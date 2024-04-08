import numpy as np
from pydrake.all import (
    AutoDiffXd,
    MultibodyPlant,
    MathematicalProgram,
    Parser,
    SnoptSolver,
    SceneGraph,
    JacobianWrtVariable,
    AddDefaultVisualization,
    DiscreteContactApproximation,
    PidController,
    RobotDiagramBuilder,
    Simulator,
    StartMeshcat,
)
from underactuated.underactuated import ConfigureParser
from underactuated.underactuated.multibody import MakePidStateProjectionMatrix

meshcat = StartMeshcat()
# meshcat = None

def run_pid_control():
    robot_builder = RobotDiagramBuilder(time_step=1e-4)

    parser = robot_builder.parser()
    ConfigureParser(parser)
    parser.AddModelsFromUrl("package://underactuated/models/spot/spot.dmd.yaml")
    parser.AddModelsFromUrl("package://underactuated/models/littledog/ground.urdf")
    plant = robot_builder.plant()
    plant.set_discrete_contact_approximation(DiscreteContactApproximation.kLagged)
    plant.Finalize()

    builder = robot_builder.builder()
    # Add a PD Controller
    plant.num_positions()
    plant.num_velocities()
    num_u = plant.num_actuators()
    kp = 150 * np.ones(num_u)
    ki = 0.0 * np.ones(num_u)
    kd = 10.0 * np.ones(num_u)
    # Select the joint states (and ignore the floating-base states)
    S = MakePidStateProjectionMatrix(plant)

    control = builder.AddSystem(
        PidController(
            kp=kp,
            ki=ki,
            kd=kd,
            state_projection=S,
            output_projection=plant.MakeActuationMatrix()[6:, :].T,
        )
    )

    builder.Connect(
        plant.get_state_output_port(), control.get_input_port_estimated_state()
    )
    builder.Connect(control.get_output_port(), plant.get_actuation_input_port())

    AddDefaultVisualization(builder, meshcat=meshcat)

    diagram = robot_builder.Build()
    simulator = Simulator(diagram)
    context = simulator.get_mutable_context()
    plant_context = plant.GetMyContextFromRoot(context)
    x0 = S @ plant.get_state_output_port().Eval(plant_context)
    control.get_input_port_desired_state().FixValue(
        control.GetMyContextFromRoot(context), x0
    )

    simulator.set_target_realtime_rate(0)
    meshcat.StartRecording()
    simulator.AdvanceTo(3.0)
    meshcat.PublishRecording()
run_pid_control()

###################################################################################################

builder = RobotDiagramBuilder(time_step=1e-4)

# parse Spot urdf and default position
parser = builder.parser()
ConfigureParser(parser)
parser.AddModelsFromUrl("package://underactuated/models/spot/spot.dmd.yaml")
parser.AddModelsFromUrl("package://underactuated/models/littledog/ground.urdf")

# create mutibody plant Spot
spot = builder.plant()
spot.set_discrete_contact_approximation(DiscreteContactApproximation.kLagged)
spot.Finalize()
builder = builder.builder() # get back diagram builder

# autodiff copy of spot
spot_ad = spot.ToAutoDiffXd()

# create contexts to work with
spot_context = spot.CreateDefaultContext()
spot_context_ad = spot.CreateDefaultContext()

# constants
nq = spot.num_positions()
nf = 3 # components for friction force (3d for spot's feet)
friction = 0.2

def manipulator_equations(vars):
    '''
    Function that given the current configuration, velocity,
    acceleration, and contact force at the stance foot, evaluates
    the manipulator equations.

    vars: concatenated q, q', q'', and friction vectors

    Returns: vector with dimensions equal to the number of configuration
    variables. If the output of this function is equal to zero
    then the given arguments verify the manipulator equations
    '''
    assert vars.size == 3*nq + 3*nf
    q, qd, qdd, f_fl, f_fr, f_bl, f_br = np.split(vars, [nq, 2*nq, 3*nq, 3*nq+nf, 3*nq+2*nf])

    # set plant accordingly
    plant, context = spot_ad, spot_context_ad if isinstance(vars[0], AutoDiffXd) else spot, spot_context
    plant = spot # TODO remove this line

    # set state
    plant.SetPositions(context, q)
    plant.SetVelocities(context, qd)

    # matrices for manipulator equations
    M = plant.CalcMassMatrixViaInverseDynamics(context)
    Cv = plant.CalcBiasTerm(context)
    tauG = plant.CalcGravityGeneralizedForces(context)

    # Jacobian of feet for contact modeling
    J_fl = calc_foot_jacobian(plant, context, )
    J_fr = calc_foot_jacobian(plant, context, )
    J_bl = calc_foot_jacobian(plant, context, )
    J_br = calc_foot_jacobian(plant, context, )

    return M@qdd + Cv - tauG - J_fl.T@f_fl - J_fr.T@f_fr - J_bl.T@f_bl - J_br.T@f_br

def calc_foot_jacobian(plant, context, foot_frame, wrt_frame, jacobian_var):
    ground_frame = spot.GetBodyByName("ground").body_frame()
    J = plant.CalcJacobianSpatialVelocity(
        context,
        JacobianWrtVariable(jacobian_var),
        foot_frame,
        wrt_frame,
        ground_frame,
        ground_frame
    )
    return J