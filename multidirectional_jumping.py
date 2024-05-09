import numpy as np
import time
from pid_standing import run_pid_control
import matplotlib.pyplot as plt
from functools import partial
from pydrake.all import (
    IpoptSolver,
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
    namedview,
)
from underactuated.underactuated import ConfigureParser

def autoDiffArrayEqual(a, b):
    return np.array_equal(a, b) and np.array_equal(ExtractGradient(a), ExtractGradient(b))


###########   INITIALIZATION   ###########
meshcat = StartMeshcat()
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



PositionView = namedview("Positions", plant.GetPositionNames(spot))
VelocityView = namedview("Velocities", plant.GetVelocityNames(spot))



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
initial_foot_positions = [
    [2.20252120e-01, 1.65945000e-01, 0],
    [2.20252120e-01, -1.65945000e-01, 0],
    [-3.75447880e-01, 1.65945000e-01, 0],
    [-3.75447880e-01, -1.65945000e-01, 0],
]

N_stance = 60
N_flight = 61
T_stance = 1
h_stance = T_stance/(N_stance-1)
max_jump_time = 2
min_jump_time = .5
N = N_stance + N_flight
in_stance = np.zeros((4, N), dtype=bool)
in_stance[:, :N_stance] = True
min_dist_above_ground = 0.1


###########   JUMP OPTIMIZATION   ###########
prog = MathematicalProgram()

##### Time steps #####
h = prog.NewContinuousVariables(N-1, "h")
prog.AddBoundingBoxConstraint(h_stance, h_stance, h[:N_stance])
prog.AddBoundingBoxConstraint(min_jump_time/N_flight, max_jump_time/N_flight, h[N_stance:])

##### Variables #####
context = [plant.CreateDefaultContext() for _ in range(N)]
q = prog.NewContinuousVariables(nq, N, "q")
v = prog.NewContinuousVariables(nv, N, "v")
CoM = prog.NewContinuousVariables(3, N, "CoM")
CoMd = prog.NewContinuousVariables(3, N, "CoMd")
CoMdd = prog.NewContinuousVariables(3, N-1, "CoMdd")
H = prog.NewContinuousVariables(3, N, "H")
Hd = prog.NewContinuousVariables(3, N-1, "Hd")
contact_force = [prog.NewContinuousVariables(3, N-1, f"foot{i}_contact_force") for i in range(4)]
# TODO try adding feet positions
q_view = PositionView(q)
v_view = VelocityView(v)

##### Guesses #####
prog.SetInitialGuess(H, np.zeros((3, N)))
prog.SetInitialGuess(Hd, np.zeros((3, N-1)))
for n in range(N):
    prog.SetInitialGuess(q[7:, n], q0[7:]) # joint positions nominal position

##### Constraints for all time #####
for n in range(N):
    # Unit quaternions
    AddUnitQuaternionConstraintOnPlant(plant, q[:, n], prog)
    # Joint position limits
    prog.AddBoundingBoxConstraint(plant.GetPositionLowerLimits(), plant.GetPositionUpperLimits(), q[:, n])
    # Joint velocity limits
    prog.AddBoundingBoxConstraint(plant.GetVelocityLowerLimits(), plant.GetVelocityUpperLimits(), v[:, n])

##### Initial state constraints #####
# Position
# prog.AddLinearEqualityConstraint(q[7:, 0], q0[7:]) # Joint positions
prog.AddBoundingBoxConstraint(min_dist_above_ground, 0.55, q[6, 0]) # Height
prog.AddLinearEqualityConstraint(q[4:6, 0], [0,0]) # x,y
prog.AddLinearEqualityConstraint(v[:, 0], np.zeros(18)) # No velocity
# prog.AddLinearEqualityConstraint(q[:4, 0], [0, 0, 0, 1])
##### Final state constraints #####
prog.AddBoundingBoxConstraint([-2,-2], [-1,-1], q[4:6, -1]) # Land inside unit box
prog.AddLinearEqualityConstraint(q[7:, -1], q0[7:]) # Joints ready to absorb impact
prog.AddLinearEqualityConstraint(q[:4, -1], [0, 0, 0, 1])
prog.AddBoundingBoxConstraint(min_dist_above_ground, .55, q[6, -1])

##### Contact force constraints #####
for n in range(N-1):
    for foot in range(4):
        # Friction pyramid
        prog.AddLinearConstraint(contact_force[foot][0, n] <= mu*contact_force[foot][2, n])
        prog.AddLinearConstraint(-contact_force[foot][0, n] <= mu*contact_force[foot][2, n])
        prog.AddLinearConstraint(contact_force[foot][1, n] <= mu*contact_force[foot][2, n])
        prog.AddLinearConstraint(-contact_force[foot][1, n] <= mu*contact_force[foot][2, n])
        # Normal force >0 if in stance 0 otherwise
        if in_stance[foot, n]:
            prog.AddBoundingBoxConstraint(0.25*total_mass*9.8, np.inf, contact_force[foot][2, n])
        else:
            prog.AddLinearEqualityConstraint(contact_force[foot][2, n], 0)

# Front and back feet should apply same upward forces
# prog.AddConstraint(eq(contact_force[0][2, :], contact_force[1][2, :]))
# prog.AddConstraint(eq(contact_force[2][2, :], contact_force[3][2, :]))


##### Center of mass constraints #####
# Translational
for n in range(N-1):
    prog.AddConstraint(eq(CoM[:,n+1], CoM[:,n] + h[n]*CoMd[:,n])) # Position
    prog.AddConstraint(eq(CoMd[:,n+1], CoMd[:,n] + h[n]*CoMdd[:,n])) # Velocity
    prog.AddConstraint(eq(total_mass*CoMdd[:,n], sum(contact_force[i][:,n] for i in range(4)) + total_mass*gravity)) # ma = Σf

# Angular Momentum
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
            torque = torque + np.cross(ad_p_WF.reshape(3) - CoM_, contact_force_[:, i]) # 𝜏 = Σ(c-p) x f
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
            torque += np.cross(p_WF.reshape(3) - CoM_, contact_force_[:, i]) # 𝜏 = Σ(c-p) x f
    return Hd_ - torque # Should be 0
for n in range(N-1):
    prog.AddConstraint(eq(H[:,n+1], H[:,n] + h[n]*Hd[:,n]))
    Fn = np.concatenate([contact_force[i][:,n] for i in range(4)])
    prog.AddConstraint(
        partial(angular_momentum_constraint, context_index=n),
        lb=[0]*3,
        ub=[0]*3,
        vars=np.concatenate((q[:, n], CoM[:, n], Hd[:, n], Fn)), # h_dot = Σ(c-p) x f = 𝜏
    )

# Make sure plant obeys CoM constraints
CoM_constraint_context = [ad_plant.CreateDefaultContext() for _ in range(N)]
def CoM_constraint(vars, context_index):
    qv_, CoM_, H_ = np.split(vars, [nq+nv, nq+nv+3])
    if isinstance(vars[0], AutoDiffXd):
        if not autoDiffArrayEqual(qv_, ad_plant.GetPositionsAndVelocities(CoM_constraint_context[context_index])):
            ad_plant.SetPositionsAndVelocities(CoM_constraint_context[context_index], qv_)
        CoM_q = ad_plant.CalcCenterOfMassPositionInWorld(CoM_constraint_context[context_index], [spot])
        H_qv = ad_plant.CalcSpatialMomentumInWorldAboutPoint(CoM_constraint_context[context_index], [spot], CoM_).rotational()
    else:
        if not np.array_equal(qv_, plant.GetPositionsAndVelocities(context[context_index])):
            plant.SetPositionsAndVelocities(context[context_index], qv_)
        CoM_q = plant.CalcCenterOfMassPositionInWorld(context[context_index], [spot])
        H_qv = plant.CalcSpatialMomentumInWorldAboutPoint(context[context_index], [spot], CoM_).rotational()
    return np.concatenate((CoM_q - CoM_, H_qv - H_)) # Should be 0
for n in range(N):
    prog.AddConstraint(
        partial(CoM_constraint, context_index=n),
        lb=[0]*6,
        ub=[0]*6,
        vars=np.concatenate((q[:,n], v[:,n], CoM[:,n], H[:,n])),
    )

##### Joint velocity dynamics constraints #####
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

##### Kinematic constraints #####
def fixed_position_constraint(vars, context_index, frame):
    q_, qn_ = np.split(vars, [nq])
    if not np.array_equal(q_, plant.GetPositions(context[context_index])):
        plant.SetPositions(context[context_index], q_)
    if not np.array_equal(qn_, plant.GetPositions(context[context_index+1])):
        plant.SetPositions(context[context_index+1], qn_)
    p_WF = plant.CalcPointsPositions(
        context[context_index],
        frame,
        foot_in_leg,
        plant.world_frame(),
    )
    p_WF_n = plant.CalcPointsPositions(
        context[context_index+1],
        frame,
        foot_in_leg,
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
        return InitializeAutoDiff(p_WF_n - p_WF, J_WF_n@ExtractGradient(qn_) - J_WF@ExtractGradient(q_))
    else:
        return p_WF_n - p_WF # Should be 0
    
for foot in range(4):
    for n in range(N):
        if in_stance[foot, n]:
            # Feet on ground
            prog.AddConstraint(
                PositionConstraint(
                    plant,
                    plant.world_frame(),
                    [-np.inf, -np.inf, 0],
                    [np.inf, np.inf, 0],
                    foot_frame[foot],
                    foot_in_leg,
                    context[n],
                ),
                q[:, n],
            )
            if n > 0 and in_stance[foot, n-1]:
                prog.AddConstraint(
                    partial(fixed_position_constraint, context_index=n-1, frame=foot_frame[foot]),
                    lb=[0]*3,
                    ub=[0]*3,
                    vars=np.concatenate((q[:, n-1], q[:, n])),
                )
        else:
            prog.AddConstraint(
                PositionConstraint(
                    plant,
                    plant.world_frame(),
                    [-np.inf, -np.inf, 0],
                    [np.inf, np.inf, np.inf],
                    foot_frame[foot],
                    foot_in_leg,
                    context[n],
                ),
                q[:, n],
            )
        # Kees don't go under ground
        prog.AddConstraint(
            PositionConstraint(
                plant,
                plant.world_frame(),
                [-np.inf, -np.inf, 0],
                [np.inf, np.inf, np.inf],
                foot_frame[foot],
                [0,0,0],
                context[n],
            ),
            q[:, n],
        )

###########   SOLVE   ###########
solver = IpoptSolver()
print("Solving")
start = time.time()
result = solver.Solve(prog)
print(result.is_success())
print("Time to solve:", time.time() - start)

###########   VISUALIZE   ###########
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
# Body
plt.figure(1)
plt.plot(t_sol, result.GetSolution(CoM[2]), label="CoM")
plt.plot(t_sol, result.GetSolution(q[6]), label="q_z")
plt.legend(loc="upper left")
plt.title("Body z position")
# Body_x
plt.figure(7)
plt.plot(t_sol, result.GetSolution(CoM[0]), label="CoM")
plt.plot(t_sol, result.GetSolution(q[4]), label="q_x")
plt.legend(loc="upper left")
plt.title("Body x position")
# Forces
plt.figure(2)
plt.plot(t_sol[:-1], result.GetSolution(contact_force[0][2]), label='FL_z')
plt.plot(t_sol[:-1], result.GetSolution(contact_force[1][2]), label='FR_z')
plt.plot(t_sol[:-1], result.GetSolution(contact_force[2][2]), label='RL_z')
plt.plot(t_sol[:-1], result.GetSolution(contact_force[3][2]), label='RR_z')
plt.legend(loc="upper left")
plt.title("Feet forces (upward component)")
# # FL
# plt.figure(3)
# plt.plot(t_sol[:-1], result.GetSolution(contact_force[0][0]), label='FL_x')
# plt.plot(t_sol[:-1], result.GetSolution(contact_force[0][1]), label='FL_y')
# plt.legend(loc="upper left")
# # FR
# plt.figure(4)
# plt.plot(t_sol[:-1], result.GetSolution(contact_force[1][0]), label='FR_x')
# plt.plot(t_sol[:-1], result.GetSolution(contact_force[1][1]), label='FR_y')
# plt.legend(loc="upper left")
# # RL
# plt.figure(5)
# plt.plot(t_sol[:-1], result.GetSolution(contact_force[2][0]), label='RL_x')
# plt.plot(t_sol[:-1], result.GetSolution(contact_force[2][1]), label='RL_y')
# plt.legend(loc="upper left")
# # RR
# plt.figure(6)
# plt.plot(t_sol[:-1], result.GetSolution(contact_force[3][0]), label='RR_x')
# plt.plot(t_sol[:-1], result.GetSolution(contact_force[3][1]), label='RR_y')
# plt.legend(loc="upper left")
plt.show()


