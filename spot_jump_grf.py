import numpy as np
import matplotlib.pyplot as plt
import time
from pid_standing import run_pid_control
from pydrake.all import (
    AutoDiffXd,
    DiscreteContactApproximation,
    RobotDiagramBuilder,
    StartMeshcat,
    plot_system_graphviz,
    MathematicalProgram,
    SnoptSolver,
    JacobianWrtVariable,
    AddUnitQuaternionConstraintOnPlant,
    eq,
    AddDefaultVisualization,
    PiecewisePolynomial,
    MeshcatVisualizer,
)
from underactuated.underactuated import ConfigureParser

# Stand
meshcat = StartMeshcat()
# run_pid_control(meshcat)

###################################################################################################

robot_builder = RobotDiagramBuilder(time_step=1e-4)
parser = robot_builder.parser()
scene_graph = robot_builder.scene_graph()
ConfigureParser(parser)
(spot,) = parser.AddModelsFromUrl("package://underactuated/models/spot/spot.dmd.yaml")
parser.AddModelsFromUrl("package://underactuated/models/littledog/ground.urdf")
plant = robot_builder.plant()
plant.set_discrete_contact_approximation(DiscreteContactApproximation.kLagged)
plant.Finalize()
plant_ad = plant.ToAutoDiffXd()
builder = robot_builder.builder()
visualizer = MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat=meshcat)

diagram = robot_builder.Build()

plant_context = plant.CreateDefaultContext()
plant_context_ad = plant_ad.CreateDefaultContext()


diagram_context = diagram.CreateDefaultContext()
ctx = plant.GetMyContextFromRoot(diagram_context)
plant.SetPositions(plant_context, plant.GetDefaultPositions())
diagram.ForcedPublish(diagram_context)



nq = plant.num_positions()
nu = plant.num_actuators()
nv = plant.num_velocities()
nf = 3 # 3d friction
q0 = plant.GetDefaultPositions()
effort_ub = plant.GetEffortUpperLimits()
effort_lb = plant.GetEffortLowerLimits()
position_ub = plant.GetPositionUpperLimits()
position_lb = plant.GetPositionLowerLimits()
velocity_ub = plant.GetVelocityUpperLimits()
velocity_lb = plant.GetVelocityLowerLimits()
accel_ub = plant.GetAccelerationUpperLimits()
accel_lb = plant.GetAccelerationLowerLimits()
h_min = 0.01
h_max = 0.1
h = 0.01
mu = 1.0

##### Jump Optimization
prog = MathematicalProgram()

N_launch = 21
N_flight = 50
N = N_launch + N_flight

q = prog.NewContinuousVariables(rows=N, cols=nq, name="q")
v = prog.NewContinuousVariables(rows=N, cols=nv, name="v")
vd = prog.NewContinuousVariables(rows=N, cols=nv, name="vd")
u = prog.NewContinuousVariables(rows=N, cols=nu, name="u")
f_fl = prog.NewContinuousVariables(rows=N, cols=nf, name="fl_friction")
f_fr = prog.NewContinuousVariables(rows=N, cols=nf, name="fr_friction")
f_rl = prog.NewContinuousVariables(rows=N, cols=nf, name="rl_friction")
f_rr = prog.NewContinuousVariables(rows=N, cols=nf, name="rr_friction")

# timestep
h = prog.NewContinuousVariables(N, name="h")
prog.AddBoundingBoxConstraint([h_min]*N, [h_max]*N, h)

# initial position
prog.AddLinearEqualityConstraint(q[0], q0)
prog.AddLinearEqualityConstraint(v[0], np.zeros_like(v[0]))

# final position
prog.AddLinearEqualityConstraint(q[-1], q0)

for n in range(N):
    # unit quaternions
    AddUnitQuaternionConstraintOnPlant(plant, q[n], prog)
    AddUnitQuaternionConstraintOnPlant(plant_ad, q[n], prog)

    # joint limits
    for j in range(nq):
        prog.AddLinearConstraint(q[n, j] >= position_lb[j])
        prog.AddLinearConstraint(q[n, j] <= position_ub[j])

    # Actuator limits
    for i in range(nu):
        prog.AddLinearConstraint(u[n, i] >= effort_lb[i])
        prog.AddLinearConstraint(u[n, i] <= effort_ub[i])

    # velocity, acceleration limits
    for i in range(nv):
        prog.AddLinearConstraint(v[n, i] >= velocity_lb[i])
        prog.AddLinearConstraint(v[n, i] <= velocity_ub[i])

        prog.AddLinearConstraint(vd[n, i] >= accel_lb[i])
        prog.AddLinearConstraint(vd[n, i] <= accel_ub[i])

for n in range(N-1):
    # velocity/accel constraints
    prog.AddConstraint(eq(q[n + 1][1:], q[n][1:] + h[n] * v[n + 1]))
    prog.AddConstraint(eq(v[n + 1], v[n] + h[n] * vd[n]))

# def continuous_centroidal_dynamics(X_dyn):
#     # returns X_dyn_dot
#     assert X_dyn.size == 3+3+


print(plant.CalcSpatialMomentumInWorldAboutPoint(plant_context, [0,0,0]))

solver = SnoptSolver()

print("Solving")
start = time.time()
result = solver.Solve(prog)
print(result.is_success())
print("Time to solve:", time.time() - start)

# visualize
print("Visualizing")
context = diagram.CreateDefaultContext()
plant_context = plant.GetMyContextFromRoot(context)
t_sol = np.cumsum(result.GetSolution(h))
q_sol = PiecewisePolynomial.FirstOrderHold(t_sol, result.GetSolution(q).T)
visualizer.StartRecording()
t0 = t_sol[0]
tf = t_sol[-1]
for t in t_sol:
    context.SetTime(t)
    plant.SetPositions(plant_context, q_sol.value(t))
    diagram.ForcedPublish(context)
visualizer.StopRecording()
visualizer.PublishRecording()
