from math import pi

class RobotConfig(object):
    urdf_path = r'assets/thmos_urdf_mix/urdf/thmos_mix.urdf'
    center_height = 0.435521
    control_freq = 30 # control frequency 50 Hz
    lower_joints = [-0.4, -2.0, -0.2, -0.1, -pi / 2.5, -0.5,  # left leg
                    0, 0, -pi, -1.6, 0, -pi, -1.6, -2.5,  # head & arm
                    -0.4, -1.8, -0.2, -1.9, -pi / 2.5, -0.5,  # right leg
                    ]
    upper_joints = [0.4, 1.8, 0.2, 1.9, pi / 2.5, 0.5,
                    0, 0, pi, 1.6, 2.5, pi, 1.6, 0,
                    0.4, 2.0, 0.2, 0.1, pi / 2.5, 0.5,
                    ]
    max_torque = 7.0
    max_velocity = 7.0
    lpf_ratio = 0.2 # low pass filter ratio 


class SimulationConfig(object):
    simulation_freq = 240 # simulation frequency 500 Hz

    # ground
    ground_lateral_friction = 1.0
    ground_spinning_friction = 0.1
    ground_rolling_friction = 0.001
    ground_restitution = 0.9

    # ball

    # goal
    goal_radius = 0.3