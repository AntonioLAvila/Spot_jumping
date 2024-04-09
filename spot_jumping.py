import numpy as np
import matplotlib.pyplot as plt
from pid_standing import run_pid_control
from pydrake.all import (
    DiscreteContactApproximation,
    RobotDiagramBuilder,
    StartMeshcat,
    plot_system_graphviz,
    DirectTranscription,
    Solve,
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
parser.AddModelsFromUrl("package://underactuated/models/spot/spot.dmd.yaml")
parser.AddModelsFromUrl("package://underactuated/models/littledog/ground.urdf")

# create mutibody plant Spot
plant = builder.plant()
plant.set_discrete_contact_approximation(DiscreteContactApproximation.kLagged)
plant.Finalize()
builder = builder.builder()

# Show diagram
plot_system_graphviz(builder.Build())
plt.show()

# create contexts to work with
plant_context = plant.CreateDefaultContext()

# constants
nq = plant.num_positions() + plant.num_velocities()
nu = plant.num_actuators()


##### Direct Transcription
N = 41
dirtran = DirectTranscription(
    plant,
    plant_context,
    num_time_samples=N,
    input_port_index=plant.GetInputPort('spot_actuation').get_index(),
)
prog = dirtran.prog()

q = dirtran.state()
q_vars = prog.decision_variables()[:N*nq]
q_vars = q_vars.reshape((41, 37))

u = dirtran.input()
u_vars = prog.decision_variables()[N*nq:]
u_vars = u_vars.reshape((41, 12))

t = dirtran.time()

for i in range(N):
    AddUnitQuaternionConstraintOnPlant(plant, q_vars[i, :plant.num_positions()], prog)

u_lower_limit = plant.GetEffortLowerLimits()
u_upper_limit = plant.GetEffortUpperLimits()
for i in range(nu):
    dirtran.AddConstraintToAllKnotPoints(u[i] <= u_upper_limit[i])
    dirtran.AddConstraintToAllKnotPoints(u[i] >= u_lower_limit[i])

result = Solve(prog)
