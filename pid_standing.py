from pydrake.all import (
    RobotDiagramBuilder,
    PidController,
    DiscreteContactApproximation,
    AddDefaultVisualization,
    Simulator,
)
import numpy as np
from underactuated.underactuated import ConfigureParser
from underactuated.underactuated.multibody import MakePidStateProjectionMatrix

def run_pid_control(meshcat):
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