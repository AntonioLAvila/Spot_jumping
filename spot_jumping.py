import numpy as np
import time
from pid_standing import run_pid_control
from functools import partial
from pydrake.all import (
    DiscreteContactApproximation,
    RobotDiagramBuilder,
    StartMeshcat,
    MathematicalProgram,
    SnoptSolver,
    AddUnitQuaternionConstraintOnPlant,
    MeshcatVisualizer,
    OrientationConstraint,
    RotationMatrix,
    AutoDiffXd,
    ExtractGradient,
    ExtractValue,
    JacobianWrtVariable,
    InitializeAutoDiff,
    PositionConstraint,
    PiecewisePolynomial,
    eq,
)
from underactuated.underactuated import ConfigureParser

def autoDiffArrayEqual(a, b):
    return np.array_equal(a, b) and np.array_equal(ExtractGradient(a), ExtractGradient(b))

meshcat = StartMeshcat()
# run_pid_control(meshcat)

###########   INITIALIZATION   ###########
robot_builder = RobotDiagramBuilder(time_step=1e-4)
plant = robot_builder.plant()
scene_graph = robot_builder.scene_graph()
parser = robot_builder.parser()
ConfigureParser(parser)
(spot,) = parser.AddModelsFromUrl("package://underactuated/models/spot/spot.dmd.yaml")
parser.AddModelsFromUrl("package://underactuated/models/littledog/ground.urdf")
plant.set_discrete_contact_approximation(DiscreteContactApproximation.kLagged)
plant.Finalize()
builder = robot_builder.builder()
visualizer = MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat=meshcat)

diagram = robot_builder.Build()
diagram_context = diagram.CreateDefaultContext()
plant_context = plant.GetMyContextFromRoot(diagram_context)
q0 = plant.GetDefaultPositions()
q0[6] -= 0.02889683
plant.SetPositions(plant_context, q0)
diagram.ForcedPublish(diagram_context)



nq = plant.num_positions()
nv = plant.num_velocities()
mu = 1.0

ad_plant = plant.ToAutoDiffXd()

body_frame = plant.GetFrameByName("body")
total_mass = plant.CalcTotalMass(plant_context, [spot])
gravity = plant.gravity_field().gravity_vector()

foot_frame = [
    plant.GetFrameByName("front_left_lower_leg"),
    plant.GetFrameByName("front_right_lower_leg"),
    plant.GetFrameByName("rear_left_lower_leg"),
    plant.GetFrameByName("rear_right_lower_leg"),
]
foot_in_leg = [0,0,-0.3365-0.036]

N_stance = 60
N_flight = 61
T_stance = 3
h_stance = T_stance/(N_stance-1)
max_jump_time = 3
min_jump_time = 0.5
N = N_stance + N_flight
in_stance = np.zeros((4, N), dtype=bool)
in_stance[:, :N_stance] = True


###########   JUMP OPTIMIZATION   ###########
prog = MathematicalProgram()

# Time steps
h = prog.NewContinuousVariables(N-1, "h")
prog.AddBoundingBoxConstraint(h_stance, h_stance, h[:N_stance])
prog.AddBoundingBoxConstraint(min_jump_time/N_flight, max_jump_time/N_flight, h[N_stance:])


context = [plant.CreateDefaultContext() for _ in range(N)]
q = prog.NewContinuousVariables(nq, N, "q")
v = prog.NewContinuousVariables(nv, N, "v")

# Constraints invariant of time
for n in range(N):
    # Unit quaternions
    AddUnitQuaternionConstraintOnPlant(plant, q[:, n], prog)
    # Joint position limits
    prog.AddBoundingBoxConstraint(plant.GetPositionLowerLimits(), plant.GetPositionUpperLimits(), q[:, n])
    # Joint velocity limits
    prog.AddBoundingBoxConstraint(plant.GetVelocityLowerLimits(), plant.GetVelocityUpperLimits(), v[:, n])
    # Initial guess is the default position
    prog.SetInitialGuess(q[:, n], q0)

# Velocity dynamics constraints
ad_velocity_dynamics_context = [ad_plant.CreateDefaultContext() for _ in range(N)]
def velocity_dynamics_constraint(vars, context_index):
    h_, q_, v_, qn_ = np.split(vars, [1, 1+nq, 1+nq+nv])
    if isinstance(vars[0], AutoDiffXd):
        if not autoDiffArrayEqual(q_, ad_plant.GetPositions(ad_velocity_dynamics_context[context_index])):
            ad_plant.SetPositions(ad_velocity_dynamics_context[context_index], q_)
        v_qd = ad_plant.MapQDotToVelocity(ad_velocity_dynamics_context[context_index], (qn_-q_)/h_)
    else:
        if not np.array_equal(q_, plant.GetPositions(context[context_index])):
            plant.SetPositions(context[context_index], q_)
        v_qd = plant.MapQDotToVelocity(context[context_index], (qn_-q_)/h_)
    return v_ - v_qd # Should be 0
for n in range(N-1):
    prog.AddConstraint(
        partial(velocity_dynamics_constraint, context_index=n),
        lb=[0]*nv,
        ub=[0]*nv,
        vars=np.concatenate(([h[n]], q[:,n], v[:,n], q[:,n+1])),
    )

# Contact force constraints
contact_force = [prog.NewContinuousVariables(3, N-1, f"foot{i}_contact_force") for i in range(4)]
for foot in range(4):
    for n in range(N-1):
        # Friction pyramid TODO change to friction cone
        prog.AddLinearConstraint(contact_force[foot][0, n] <= mu*contact_force[foot][2, n])
        prog.AddLinearConstraint(contact_force[foot][0, n] >= -mu*contact_force[foot][2, n])
        prog.AddLinearConstraint(contact_force[foot][1, n] <= mu*contact_force[foot][2, n])
        prog.AddLinearConstraint(contact_force[foot][1, n] >= -mu*contact_force[foot][2, n])
        # Normal force >=0 if in stance 0 otherwise
        if in_stance[foot, n]:
            prog.AddBoundingBoxConstraint(0, np.inf, contact_force[foot][2, n])
            # prog.AddBoundingBoxConstraint(0, 4*9.81*total_mass, contact_force[foot][2, n])
        else:
            prog.AddBoundingBoxConstraint(0, 0, contact_force[foot][2, n])


# Center of mass translational constraints
CoM = prog.NewContinuousVariables(3, N, "CoM")
CoMd = prog.NewContinuousVariables(3, N, "CoMd")
CoMdd = prog.NewContinuousVariables(3, N-1, "CoMdd")
prog.AddBoundingBoxConstraint(q0[4:7], q0[4:7], CoM[:, 0]) # Initial CoM position = q0
prog.AddBoundingBoxConstraint(0, 0, CoMd[:, 0]) # Initial CoM vel = 0
prog.AddBoundingBoxConstraint([0, 0], [0, 0], CoM[:2, -1]) # Final CoM position = q0
for n in range(N): # initial guess is parabola
    if n < N_stance:
        prog.SetInitialGuess(CoM[:, n], q0[4:7])
    else:
        prog.SetInitialGuess(CoM[:, n], [0,0,-(n-N_flight)*(n-N-1)])

# CoM dynamics
for n in range(N-1):
    prog.AddConstraint(eq(CoM[:,n+1], CoM[:,n] + h[n]*CoMd[:,n])) # Position
    prog.AddConstraint(eq(CoMd[:,n+1], CoMd[:,n] + h[n]*CoMdd[:,n])) # Velocity
    prog.AddConstraint(eq(total_mass*CoMdd[:,n], sum(contact_force[i][:,n] for i in range(4)) + total_mass*gravity)) # ma = Σf + fg

# Center of mass angular constraints
H = prog.NewContinuousVariables(3, N, "H")
Hd = prog.NewContinuousVariables(3, N-1, "Hdot")
prog.SetInitialGuess(H, np.zeros((3, N))) # Start unturned
prog.SetInitialGuess(Hd, np.zeros((3, N-1))) # Start not spinning
def angular_momentum_constraint(vars, context_index):
    q_, CoM_, Hd_, contact_force_ = np.split(vars, [nq, 3+nq, 6+nq])
    contact_force_ = contact_force_.reshape(3, 4, order="F")
    if isinstance(vars[0], AutoDiffXd):
        dq = ExtractGradient(q_)
        q_ = ExtractValue(q_)
        if not np.array_equal(q_, plant.GetPositions(context[context_index])):
            plant.SetPositions(context[context_index], q_)
        torque = np.zeros(3)
        for i in range(4):
            p_WF = plant.CalcPointsPositions(
                context[context_index],
                foot_frame[i],
                foot_in_leg,
                plant.world_frame(),
            )
            Jq_WF = plant.CalcJacobianTranslationalVelocity(
                context[context_index],
                JacobianWrtVariable.kQDot,
                foot_frame[i],
                foot_in_leg,
                plant.world_frame(),
                plant.world_frame(),
            )
            ad_p_WF = InitializeAutoDiff(p_WF, Jq_WF@dq)
            torque = torque + np.cross(ad_p_WF.reshape(3) - CoM_, contact_force_[:, i]) # h_dot = (c-p)xf
    else:
        if not np.array_equal(q_, plant.GetPositions(context[context_index])):
            plant.SetPositions(context[context_index], q_)
        torque = np.zeros(3)
        for i in range(4):
            p_WF = plant.CalcPointsPositions(
                context[context_index],
                foot_frame[i],
                foot_in_leg,
                plant.world_frame(),
            )
            torque += np.cross(p_WF.reshape(3) - CoM_, contact_force_[:, i]) # h_dot = (c-p)xf
    return Hd_ - torque # Should be 0
for n in range(N-1):
    prog.AddConstraint(eq(H[:,n+1], H[:,n] + h[n]*Hd[:,n]))
    Fn = np.concatenate([contact_force[i][:,n] for i in range(4)])
    prog.AddConstraint(
        partial(angular_momentum_constraint, context_index=n),
        lb=[0]*3,
        ub=[0]*3,
        vars=np.concatenate((q[:, n], CoM[:, n], Hd[:, n], Fn)), # h_dot = (c-p)xf
    )

# Make sure plant obeys CoM constraints
CoM_constraint_context = [ad_plant.CreateDefaultContext() for _ in range(N)]
def CoM_constraint(vars, context_index):
    qv, CoM, H = np.split(vars, [nq+nv, nq+nv+3])
    if isinstance(vars[0], AutoDiffXd):
        if not autoDiffArrayEqual(qv, ad_plant.GetPositionsAndVelocities(CoM_constraint_context[context_index])):
            ad_plant.SetPositionsAndVelocities(CoM_constraint_context[context_index], qv)
        CoM_q = ad_plant.CalcCenterOfMassPositionInWorld(CoM_constraint_context[context_index], [spot])
        H_qv = ad_plant.CalcSpatialMomentumInWorldAboutPoint(CoM_constraint_context[context_index], [spot], CoM).rotational()
    else:
        if not np.array_equal(qv, plant.GetPositionsAndVelocities(CoM_constraint_context[context_index])):
            plant.SetPositionsAndVelocities(CoM_constraint_context[context_index], qv)
        CoM_q = plant.CalcCenterOfMassPositionInWorld(context[context_index], [spot])
        H_qv = plant.CalcSpatialMomentumInWorldAboutPoint(CoM_constraint_context[context_index], [spot], CoM).rotational()
    return np.concatenate((CoM_q - CoM, H_qv - H)) # Should be [0,0,0,0,0,0]
for n in range(N):
    prog.AddConstraint(
        partial(CoM_constraint, context_index=n),
        lb=[0]*6,
        ub=[0]*6,
        vars=np.concatenate((q[:,n], v[:,n], CoM[:,n], H[:,n])),
    )

# Kinematic constraints
def fixed_position_constraint(vars, context_index, frame):
    q, qn = np.split(vars, [nq])
    if not np.array_equal(q, plant.GetPositions(context[context_index])):
        plant.SetPositions(context[context_index], q)
    if not np.array_equal(qn, plant.GetPositions(context[context_index+1])):
        plant.SetPositions(context[context_index+1], qn)
    p_WF = plant.CalcPointsPositions(
        context[context_index],
        frame,
        [0, 0, 0],
        plant.world_frame(),
    )
    p_WF_n = plant.CalcPointsPositions(
        context[context_index+1],
        frame,
        [0, 0, 0],
        plant.world_frame(),
    )
    if isinstance(vars[0], AutoDiffXd):
        J_WF = plant.CalcJacobianTranslationalVelocity(
            context[context_index],
            JacobianWrtVariable.kQDot,
            frame,
            foot_in_leg,
            plant.world_frame(),
            plant.world_frame(),   
        )
        J_WF_n = plant.CalcJacobianTranslationalVelocity(
            context[context_index+1],
            JacobianWrtVariable.kQDot,
            frame,
            foot_in_leg,
            plant.world_frame(),
            plant.world_frame(),   
        )
        return InitializeAutoDiff(p_WF_n - p_WF, J_WF_n@ExtractGradient(qn) - J_WF@ExtractGradient(q))
    else:
        return p_WF_n - p_WF # Should be 0

for i in range(4):
    for n in range(N):
        if in_stance[i, n]:
            # Feet on ground
            prog.AddConstraint(
                PositionConstraint(
                    plant,
                    plant.world_frame(),
                    [-np.inf, -np.inf, 0],
                    [np.inf, np.inf, 0],
                    foot_frame[i],
                    foot_in_leg,
                    context[n],
                ),
                q[:, n],
            )
            # Feet on ground don't move
            if n>0 and in_stance[i, n-1]:
                prog.AddConstraint(
                    partial(fixed_position_constraint, context_index=n-1, frame=foot_frame[i]),
                    lb=[0]*3,
                    ub=[0]*3,
                    vars=np.concatenate((q[:,n-1], q[:,n])),
                )
        else:
            # Feet somewhere off ground
            prog.AddConstraint(
                PositionConstraint(
                    plant,
                    plant.world_frame(),
                    [-np.inf, -np.inf, 1e-4],
                    [np.inf, np.inf, np.inf],
                    foot_frame[i],
                    foot_in_leg,
                    context[n],
                ),
                q[:, n],
            )

###########   SOLVE   ###########
solver = SnoptSolver()
print("Solving")
start = time.time()
result = solver.Solve(prog)
print(result.is_success())
print("Time to solve:", time.time() - start)

###########   VISUALIZE   ###########
print(result.GetSolution(h))
print("Visualizing")
context = diagram.CreateDefaultContext()
plant_context = plant.GetMyContextFromRoot(context)
t_sol = np.cumsum(np.concatenate(([0], result.GetSolution(h))))
q_sol = PiecewisePolynomial.FirstOrderHold(t_sol, result.GetSolution(q))
visualizer.StartRecording()
t0 = t_sol[0]
tf = t_sol[-1]
for t in t_sol:
    context.SetTime(t)
    plant.SetPositions(plant_context, q_sol.value(t))
    diagram.ForcedPublish(context)
visualizer.StopRecording()
visualizer.PublishRecording()
# while True: pass # Keep the viz up


