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
fl_friction = prog.NewContinuousVariables(rows=N, cols=nf, name="fl_friction")
fr_friction = prog.NewContinuousVariables(rows=N, cols=nf, name="fr_friction")
rl_friction = prog.NewContinuousVariables(rows=N, cols=nf, name="rl_friction")
rr_friction = prog.NewContinuousVariables(rows=N, cols=nf, name="rr_friction")

# timestep
h = prog.NewContinuousVariables(N, name="h")
prog.AddBoundingBoxConstraint([h_min] * N, [h_max] * N, h)

# Manipulator Equations
def manipulator_equations(vars):
    assert vars.size == nq +  2*nv + 4*nf
    m_q, m_v, m_vd, f_fl, f_fr, f_rl, f_rr = np.split(vars, [
        nq, 
        nq + nv, 
        nq + 2*nv, 
        nq + 2*nv + nf, 
        nq + 2*nv + 2*nf, 
        nq + 2*nv + 3*nf,
    ])

    # set plant accordingly
    m_plant, m_context = (plant_ad, plant_context_ad) if isinstance(vars[0], AutoDiffXd) else (plant, plant_context)

    # set state
    m_plant.SetPositions(m_context, m_q)
    m_plant.SetVelocities(m_context, m_v)

    # matrices for manipulator equations
    M = m_plant.CalcMassMatrixViaInverseDynamics(m_context)
    Cv = m_plant.CalcBiasTerm(m_context)
    tauG = m_plant.CalcGravityGeneralizedForces(m_context)

    # Jacobian of feet for contact
    world_frame = m_plant.GetFrameByName("ground")
    J_fl = calc_foot_jacobian(m_plant, m_context, m_plant.GetFrameByName("front_left_lower_leg"), world_frame, [0,0,0])
    J_fr = calc_foot_jacobian(m_plant, m_context, m_plant.GetFrameByName("front_right_lower_leg"), world_frame, [0,0,0])
    J_rl = calc_foot_jacobian(m_plant, m_context, m_plant.GetFrameByName("rear_left_lower_leg"), world_frame, [0,0,0])
    J_rr = calc_foot_jacobian(m_plant, m_context, m_plant.GetFrameByName("rear_right_lower_leg"), world_frame, [0,0,0])

    return M@m_vd + Cv - tauG - J_fl.T@f_fl - J_fr.T@f_fr - J_rl.T@f_rl - J_rr.T@f_rr # should be equal to 0

def calc_foot_jacobian(plt, ctxt, foot_frame, world_frame, position_in_frame):
    J = plt.CalcJacobianTranslationalVelocity(
        ctxt,
        JacobianWrtVariable.kV,
        foot_frame,
        position_in_frame,
        world_frame,
        world_frame,
    )
    return J

# initial position
prog.AddLinearEqualityConstraint(q[0], q0)
prog.AddLinearEqualityConstraint(v[0], np.zeros_like(v[0]))

# final position
prog.AddLinearEqualityConstraint(q[-1], q0)

# feet forces
# for n in range(N_launch):
#     prog.AddLinearConstraint(fl_friction[n, 2] >= 0)
#     prog.AddLinearConstraint(fr_friction[n, 2] >= 0)
#     prog.AddLinearConstraint(rl_friction[n, 2] >= 0)
#     prog.AddLinearConstraint(rr_friction[n, 2] >= 0)

#     prog.AddLinearConstraint(fl_friction[n, 0] <= mu*fl_friction[n, 2])
#     prog.AddLinearConstraint(fl_friction[n, 0] >= -mu*fl_friction[n, 2])
#     prog.AddLinearConstraint(fl_friction[n, 1] <= mu*fl_friction[n, 2])
#     prog.AddLinearConstraint(fl_friction[n, 1] >= -mu*fl_friction[n, 2])

#     prog.AddLinearConstraint(fr_friction[n, 0] <= mu*fr_friction[n, 2])
#     prog.AddLinearConstraint(fr_friction[n, 0] >= -mu*fr_friction[n, 2])
#     prog.AddLinearConstraint(fr_friction[n, 1] <= mu*fr_friction[n, 2])
#     prog.AddLinearConstraint(fr_friction[n, 1] >= -mu*fr_friction[n, 2])

#     prog.AddLinearConstraint(rl_friction[n, 0] <= mu*rl_friction[n, 2])
#     prog.AddLinearConstraint(rl_friction[n, 0] >= -mu*rl_friction[n, 2])
#     prog.AddLinearConstraint(rl_friction[n, 1] <= mu*rl_friction[n, 2])
#     prog.AddLinearConstraint(rl_friction[n, 1] >= -mu*rl_friction[n, 2])

#     prog.AddLinearConstraint(rr_friction[n, 0] <= mu*rr_friction[n, 2])
#     prog.AddLinearConstraint(rr_friction[n, 0] >= -mu*rr_friction[n, 2])
#     prog.AddLinearConstraint(rr_friction[n, 1] <= mu*rr_friction[n, 2])
#     prog.AddLinearConstraint(rr_friction[n, 1] >= -mu*rr_friction[n, 2])

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

    # manipulator eqations
    vars = np.concatenate((q[n+1], v[n+1], vd[n], fl_friction[n], fr_friction[n], rl_friction[n], rr_friction[n]))
    prog.AddConstraint(manipulator_equations, lb=[0]*nv, ub=[0]*nv, vars=vars)


def feet_height(q):
    m_plant, m_context = (plant_ad, plant_context_ad) if isinstance(q[0], AutoDiffXd) else (plant, plant_context)
    ground_frame = m_plant.GetBodyByName("ground").body_frame()
    m_plant.SetPositions(m_context, q)

    fl = m_plant.CalcPointsPositions(m_context, m_plant.GetBodyByName("front_left_lower_leg").body_frame(), [0,0,0], ground_frame)
    fr = m_plant.CalcPointsPositions(m_context, m_plant.GetBodyByName("front_right_lower_leg").body_frame(), [0,0,0], ground_frame)
    rl = m_plant.CalcPointsPositions(m_context, m_plant.GetBodyByName("rear_left_lower_leg").body_frame(), [0,0,0], ground_frame)
    rr = m_plant.CalcPointsPositions(m_context, m_plant.GetBodyByName("rear_right_lower_leg").body_frame(), [0,0,0], ground_frame)

    return np.array([fl[-1], fr[-1], rl[-1], rr[-1]])

for n in range(N_launch):
    prog.AddConstraint(feet_height, lb=[0]*4, ub=[np.inf]*4, vars=q[n])


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
