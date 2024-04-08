import numpy as np
from pydrake.all import (
    AutoDiffXd,
    MathematicalProgram,
    DiscreteContactApproximation,
    RobotDiagramBuilder,
    StartMeshcat,
    namedview,
    AddUnitQuaternionConstraintOnPlant,
    OrientationConstraint,
    RotationMatrix,
)
from underactuated.underactuated import ConfigureParser

meshcat = StartMeshcat()

builder = RobotDiagramBuilder(time_step=1e-4)

# parse Spot urdf and default position
parser = builder.parser()
ConfigureParser(parser)
(spot,) = parser.AddModelsFromUrl("package://underactuated/models/spot/spot.dmd.yaml")
(ground,) = parser.AddModelsFromUrl("package://underactuated/models/littledog/ground.urdf")

# create mutibody plant Spot
plant = builder.plant()
plant.set_discrete_contact_approximation(DiscreteContactApproximation.kLagged)
plant.Finalize()
builder = builder.builder() # get back diagram builder
plant_context = plant.CreateDefaultContext()
ad_plant = plant.ToAutoDiffXd()

# constants
q0 = plant.GetPositions(plant_context)
body_frame = plant.GetFrameByName("body")
# ['spot_body_qw', 'spot_body_qx', 'spot_body_qy', 'spot_body_qz', 'spot_body_x', 'spot_body_y', 'spot_body_z',
# 'spot_front_left_hip_x_q', 'spot_front_left_hip_y_q', 'spot_front_left_knee_q', 'spot_front_right_hip_x_q',
# 'spot_front_right_hip_y_q', 'spot_front_right_knee_q', 'spot_rear_left_hip_x_q', 'spot_rear_left_hip_y_q',
# 'spot_rear_left_knee_q', 'spot_rear_right_hip_x_q', 'spot_rear_right_hip_y_q', 'spot_rear_right_knee_q']
PositionView = namedview("Positions", plant.GetPositionNames(spot, always_add_suffix=False))
# ['body_wx', 'body_wy', 'body_wz', 'body_vx', 'body_vy', 'body_vz', 'front_left_hip_x', 'front_left_hip_y',
# 'front_left_knee', 'front_right_hip_x', 'front_right_hip_y', 'front_right_knee', 'rear_left_hip_x',
# 'rear_left_hip_y', 'rear_left_knee', 'rear_right_hip_x', 'rear_right_hip_y', 'rear_right_knee']
VelocityView = namedview("Velocities", plant.GetVelocityNames(spot, always_add_suffix=False))
mu = 1
total_mass = plant.CalcTotalMass(plant_context, [spot])
gravity = plant.gravity_field().gravity_vector()
nq = plant.num_positions()
nv = plant.num_velocities()
foot_frame = [
    plant.GetFrameByName("front_left_lower_leg"),
    plant.GetFrameByName("front_right_lower_leg"),
    plant.GetFrameByName("rear_left_lower_leg"),
    plant.GetFrameByName("rear_right_lower_leg"),
]


# jumping trajectory optimization
N = 41
h_min = 0.005
h_max = 0.05

prog = MathematicalProgram()

# time
h = prog.NewContinuousVariables(N, name="h")
prog.AddBoundingBoxConstraint([h_min] *N, [h_max]*N, h)

context = [plant.CreateDefaultContext() for _ in range(T)]

q = prog.NewContinuousVariables(nq, N, "q")
v = prog.NewContinuousVariables(nv, N, "v")

q_view = PositionView(q)
v_view = VelocityView(v)
q0_view = PositionView(q0)

for n in range(N):
    # joint position limits
    prog.AddBoundingBoxConstraint(
        plant.GetPositionLowerLimits(),
        plant.GetPositionUpperLimits(),
        q[:, n],
    )
    # joint velocity limits
    prog.AddBoundingBoxConstraint(
        plant.GetVelocityLowerLimits(),
        plant.GetVelocityUpperLimits(),
        v[:, n],
    )
    # unit quaternions
    AddUnitQuaternionConstraintOnPlant(plant, q[:, n], prog)
    # body orientation
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
    # start with initial guess just standing still
    prog.SetInitialGuess(q[:, n], q0)