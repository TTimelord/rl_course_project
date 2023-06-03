from math import pi

class RobotConfig(object):
    urdf_path = r'assets/thmos_urdf_mix/urdf/thmos_mix.urdf'
    center_height = 0.435521
    control_freq = 30 # control frequency 50 Hz
    # lower_joints = [-0.4, -2.0, -0.2, -0.1, -pi / 2.5, -0.5,  # left leg
    #                 0, 0, -pi, -1.6, 0, -pi, -1.6, -2.5,  # head & arm
    #                 -0.4, -1.8, -0.2, -1.9, -pi / 2.5, -0.5,  # right leg
    #                 ]
    # upper_joints = [0.4, 1.8, 0.2, 1.9, pi / 2.5, 0.5,
    #                 0, 0, pi, 1.6, 2.5, pi, 1.6, 0,
    #                 0.4, 2.0, 0.2, 0.1, pi / 2.5, 0.5,
    #                 ]
    lower_joints = [-0.4, -2.0, -0.2, -0.1, -pi / 2.5, -0.5,  # left leg
                    0, 0, -pi, -0.5, 0, -pi, -1.6, -2.5,  # head & arm
                    -0.4, -1.8, -0.2, -1.9, -pi / 2.5, -0.5,  # right leg
                    ]
    upper_joints = [0.4, 1.8, 0.2, 1.9, pi / 2.5, 0.5,
                    0, 0, pi, 0.5, 2.5, pi, 1.6, 0,
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
    
# 顺序
# 0 b'L_leg_1' 0 (-3.14, 3.14, 7.0, 8.0, b'L_leg_1_link')
# 1 b'neck' 0 (-3.14, 3.14, 3.0, 6.0, b'neck_link')
# 2 b'L_arm_1' 0 (-3.14, 3.14, 3.0, 6.0, b'L_arm_1_link')
# 3 b'R_arm_1' 0 (-3.14, 3.14, 3.0, 6.0, b'R_arm_1_link')
# 4 b'R_leg_1' 0 (-3.14, 3.14, 7.0, 8.0, b'R_leg_1_link')
# 5 b'L_leg_2' 0 (-3.14, 3.14, 7.0, 8.0, b'L_leg_2_link')
# 6 b'L_leg_3' 0 (-3.14, 3.14, 7.0, 8.0, b'L_leg_3_link')
# 7 b'L_leg_4' 0 (-0.8, 3.14, 7.0, 8.0, b'L_leg_4_link')
# 8 b'L_leg_5' 0 (-1.57, 1.57, 7.0, 8.0, b'L_leg_5_link')
# 9 b'L_leg_6' 0 (-3.14, 3.14, 7.0, 8.0, b'L_leg_6_link')
# 10 b'head' 0 (-3.14, 3.14, 3.0, 6.0, b'head_link')
# 11 b'L_arm_2' 0 (-3.14, 3.14, 3.0, 6.0, b'L_arm_2_link')
# 12 b'L_arm_3' 0 (-3.14, 3.14, 3.0, 6.0, b'L_arm_3_link')
# 13 b'R_arm_2' 0 (-3.14, 3.14, 3.0, 6.0, b'R_arm_2_link')
# 14 b'R_arm_3' 0 (-3.14, 3.14, 3.0, 6.0, b'R_arm_3_link')
# 15 b'R_leg_2' 0 (-3.14, 3.14, 7.0, 8.0, b'R_leg_2_link')
# 16 b'R_leg_3' 0 (-3.14, 3.14, 7.0, 8.0, b'R_leg_3_link')
# 17 b'R_leg_4' 0 (-3.14, 0.8, 7.0, 8.0, b'R_leg_4_link')
# 18 b'R_leg_5' 0 (-1.57, 1.57, 7.0, 8.0, b'R_leg_5_link')
# 19 b'R_leg_6' 0 (-1.57, 1.57, 7.0, 8.0, b'R_leg_6_link')


