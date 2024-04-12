import numpy as np
import time
from pid_standing import run_pid_control
from functools import partial
from pydrake.all import (
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
# plant.set_discrete_contact_approximation(DiscreteContactApproximation.kLagged)
plant.Finalize()
builder = robot_builder.builder()
visualizer = MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat=meshcat)

diagram = robot_builder.Build()
diagram_context = diagram.CreateDefaultContext()
plant_context = plant.GetMyContextFromRoot(diagram_context)
plant.SetPositions(plant_context, plant.GetDefaultPositions())
diagram.ForcedPublish(diagram_context)



nq = plant.num_positions()
nv = plant.num_velocities()
q0 = plant.GetDefaultPositions()
h_min = 0.01
h_max = 0.1
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

N_windup = 50
N = 201
in_stance = np.zeros((4, N))
in_stance[:, :N_windup]


###########   JUMP OPTIMIZATION   ###########
prog = MathematicalProgram()

# Time steps
h = prog.NewContinuousVariables(N-1, "h")
prog.AddBoundingBoxConstraint(h_min, h_max, h)

context = [plant.CreateDefaultContext() for _ in range(N)]  # Create one context per time step (to maximize cache hits)
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
    # Body orientation
    prog.AddConstraint(
        OrientationConstraint(
            plant,
            body_frame,
            RotationMatrix(),
            plant.world_frame(),
            RotationMatrix(),
            0.1,
            context[n],
        ),
        q[:, n],
    )
    # Initial guess is the default position
    prog.SetInitialGuess(q[:, n], q0)

# Velocity dynamics constraints
ad_velocity_dynamics_context = [ad_plant.CreateDefaultContext() for _ in range(N)] # Make a new autodiff context for this constraint (to maximize cache hits)
def velocity_dynamics_constraint(vars, context_index):
    h, q, v, qn = np.split(vars, [1, 1+nq, 1+nq+nv])
    if isinstance(vars[0], AutoDiffXd):
        if not autoDiffArrayEqual(q, ad_plant.GetPositions(ad_velocity_dynamics_context[context_index])):
            ad_plant.SetPositions(ad_velocity_dynamics_context[context_index], q)
        v_qd = ad_plant.MapQDotToVelocity(ad_velocity_dynamics_context[context_index], (qn-q)/h)
    else:
        if not np.array_equal(q, plant.GetPositions(context[context_index])):
            plant.SetPositions(context[context_index], q)
        v_qd = plant.MapQDotToVelocity(context[context_index], (qn-q)/h)
    return v - v_qd # Should be 0
for n in range(N-1):
    prog.AddConstraint(
        partial(velocity_dynamics_constraint, context_index=n),
        lb=[0]*nv,
        ub=[0]*nv,
        vars=np.concatenate((h[n], q[:,n], v[:,n], q[:,n+1])),
    )

# Contact force constraints
contact_force = [prog.NewContinuousVariables(3, N-1, f"foot{i}_contact_force") for i in range(4)]
for n in range(N-1):
    for foot in range(4):
        f_normal = contact_force[foot][2, n]
        # Friction pyramid TODO change to friction cone
        prog.AddLinearConstraint(contact_force[foot][0, n] <= mu*f_normal)
        prog.AddLinearConstraint(contact_force[foot][0, n] >= -mu*f_normal)
        prog.AddLinearConstraint(contact_force[foot][1, n] <= mu*f_normal)
        prog.AddLinearConstraint(contact_force[foot][1, n] >= -mu*f_normal)
        # Normal force >=0 if in stance 0 otherwise
        prog.AddBoundingBoxConstraint(0, in_stance[foot, n] * 4 * 9.81 * total_mass, f_normal)

# Center of mass translational constraints
CoM = prog.NewContinuousVariables(3, N, "CoM")
CoMd = prog.NewContinuousVariables(3, N, "CoMd")
CoMdd = prog.NewContinuousVariables(3, N-1, "CoMdd")



# ###########   SOLVE   ###########
# solver = SnoptSolver()
# print("Solving")
# start = time.time()
# result = solver.Solve(prog)
# print(result.is_success())
# print("Time to solve:", time.time() - start)

# ###########   VISUALIZE   ###########
# print("Visualizing")
# context = diagram.CreateDefaultContext()
# plant_context = plant.GetMyContextFromRoot(context)
# t_sol = np.cumsum(result.GetSolution(h))
# q_sol = PiecewisePolynomial.FirstOrderHold(t_sol, result.GetSolution(q).T)
# visualizer.StartRecording()
# t0 = t_sol[0]
# tf = t_sol[-1]
# for t in t_sol:
#     context.SetTime(t)
#     plant.SetPositions(plant_context, q_sol.value(t))
#     diagram.ForcedPublish(context)
# visualizer.StopRecording()
# visualizer.PublishRecording()



