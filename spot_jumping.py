import numpy as np
import matplotlib.pyplot as plt
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
)
from underactuated.underactuated import ConfigureParser

# Stand
# meshcat = StartMeshcat()
# run_pid_control(meshcat)

###################################################################################################

builder = RobotDiagramBuilder(time_step=1e-4)

# parse Spot urdf and default position
parser = builder.parser()
ConfigureParser(parser)
(spot,) = parser.AddModelsFromUrl("package://underactuated/models/spot/spot.dmd.yaml")
parser.AddModelsFromUrl("package://underactuated/models/littledog/ground.urdf")
plant = builder.plant()
plant.set_discrete_contact_approximation(DiscreteContactApproximation.kLagged)
plant.Finalize()
plant_ad = plant.ToAutoDiffXd()

# create contexts to work with
plant_context = plant.CreateDefaultContext()
plant_context_ad = plant_ad.CreateDefaultContext()


nq = plant.num_positions()
nu = plant.num_actuators()
nv = plant.num_velocities()
nf = 3 # 3d friction
q0 = plant.GetDefaultPositions()


##### Jump Optimization
prog = MathematicalProgram()

N_launch = 101
N_flight = 200
N = N_launch + N_flight

q = prog.NewContinuousVariables(rows=N, cols=nq, name="q")
v = prog.NewContinuousVariables(rows=N, cols=nv, name="v")
q_ddot = prog.NewContinuousVariables(rows=N, cols=nv, name="q_ddot")
u = prog.NewContinuousVariables(rows=N, cols=nu, name="u")
fl_friction = prog.NewContinuousVariables(rows=N, cols=nf, name="fl_friction")
fr_friction = prog.NewContinuousVariables(rows=N, cols=nf, name="fr_friction")
rl_friction = prog.NewContinuousVariables(rows=N, cols=nf, name="rl_friction")
rr_friction = prog.NewContinuousVariables(rows=N, cols=nf, name="rr_friction")


# Manipulator Equations
def manipulator_equations(vars):
    assert vars.size == nq +  2*nv + 4*nf
    q, v, qdd, f_fl, f_fr, f_rl, f_rr = np.split(vars, [
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
    m_plant.SetPositions(m_context, q)
    m_plant.SetVelocities(m_context, v)

    # matrices for manipulator equations
    M = m_plant.CalcMassMatrixViaInverseDynamics(m_context)
    Cv = m_plant.CalcBiasTerm(m_context)
    tauG = m_plant.CalcGravityGeneralizedForces(m_context)

    # Jacobian of feet for contact modeling
    world_frame = m_plant.GetFrameByName("ground")
    J_fl = calc_foot_jacobian(m_plant, m_context, m_plant.GetFrameByName("front_left_lower_leg"), world_frame, [0,0,0])
    J_fr = calc_foot_jacobian(m_plant, m_context, m_plant.GetFrameByName("front_right_lower_leg"), world_frame, [0,0,0])
    J_rl = calc_foot_jacobian(m_plant, m_context, m_plant.GetFrameByName("rear_left_lower_leg"), world_frame, [0,0,0])
    J_rr = calc_foot_jacobian(m_plant, m_context, m_plant.GetFrameByName("rear_right_lower_leg"), world_frame, [0,0,0])

    return M@qdd + Cv - tauG - (J_fl.T@f_fl)[1:] - (J_fr.T@f_fr)[1:] - (J_rl.T@f_rl)[1:] - (J_rr.T@f_rr)[1:] # should be equal to 0

def calc_foot_jacobian(plt, ctxt, foot_frame, world_frame, position_in_frame):
    J = plt.CalcJacobianTranslationalVelocity(
        ctxt,
        JacobianWrtVariable.kQDot,
        foot_frame,
        position_in_frame,
        world_frame,
        world_frame,
    )
    return J

for n in range(N):
    vars = np.concatenate((q[n], v[n], q_ddot[n], fl_friction[n], fr_friction[n], rl_friction[n], rr_friction[n]))
    prog.AddConstraint(manipulator_equations, lb=[0]*nv, ub=[0]*nv, vars=vars)

    AddUnitQuaternionConstraintOnPlant(plant, q[n, :], prog)
    AddUnitQuaternionConstraintOnPlant(plant_ad, q[n, :], prog)

    prog.AddLinearEqualityConstraint(q[n], q0)



solver = SnoptSolver()
result = solver.Solve(prog)
print(result.is_success())
print(result.GetSolution(q))