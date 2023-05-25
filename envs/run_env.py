from gym import error, spaces, utils
import pybullet as p
from math import pi
import numpy as np
import time
import random
from envs.kick_ball_env import KickBall
from utils import get_gravity_vec, get_omega_imu, rbf_reward

class Run(KickBall):
    def __init__(self, connect_GUI=False):
        super(Run, self).__init__(connect_GUI)
        lower_joints = self.robot_config.lower_joints
        upper_joints = self.robot_config.upper_joints
        self.observation_space = spaces.Box(low=np.array(lower_joints  + [-10]*3+[-1]*3 + [-3]*4),  # joints, omega, grav, goal_pos
                                            high=np.array(upper_joints + [10]*3 + [1]*3 + [3]*4))

    def step(self, action):
        '''filter action and step simulation'''
        new_action = np.array(action)
        real_action = self.robot_config.lpf_ratio * new_action + (1 - self.robot_config.lpf_ratio) * self.last_action
        self.last_action = real_action
        self.set_qpos_all(real_action)
        for s in range(self.sim_per_control):
            p.stepSimulation()
            self.simstep_cnt += 1
            if self.gui:
                time.sleep(1/600.)  # slow down

        '''compute observation'''
        pos, vel, omega, grav, qpos, qvel, react, torq, ball_pos, ball_velocity, goal_pos = self.get_state()
        ball_pos_relative, goal_pos_relative = self.compute_relative_pos(pos, ball_pos, goal_pos)
        observation = qpos + omega + grav + vel[:2] + goal_pos_relative

        '''compute reward'''
        ori_reward = 1.0 * rbf_reward(grav, [0, 0, -1], -1.)
        height = pos[2]
        height_reward = 1.0 * rbf_reward(height, self.robot_config.center_height * 0.999, -10.)
        # omega_reward = 0.067 * rbf_reward(omega, [0, 0, 0], -0.05)
        # joint_torq_regu_reward = 0.067 * rbf_reward(torq, 0, -0.5)
        # joint_velo_regu_reward = 0.067 * rbf_reward(qvel, 0, -0.05)
        # foot_contact, body_contact = self.contact_reward()
        # foot_contact_reward = 0.033 * foot_contact
        # body_contact_reward = 0.067 * body_contact
        # dists = [c[8] for c in p.getClosestPoints(self.robotID, self.planeId, 1.)]
        # dist = min(dists + [1.])
        # nojump_reward = 0.033 * rbf_reward(dist, 0, -100.0)
        velocity_reward = 2 * self.compute_relative_velocity(vel, pos, goal_pos)
        goal_distance = np.linalg.norm(np.array(pos)[:2] - np.array(goal_pos), ord=2)
        goal_reward = 1000 if goal_distance < self.sim_config.goal_radius else 0

        # reward = ori_reward + height_reward + omega_reward + \
        #          joint_torq_regu_reward + joint_velo_regu_reward + \
        #          foot_contact_reward + body_contact_reward + nojump_reward

        reward = velocity_reward + goal_reward
        info = {
                'ori_reward': ori_reward, 'height_reward': height_reward,
                # 'omega_reward': omega_reward, 'torq_regu': joint_torq_regu_reward,
                # 'velo_regu': joint_velo_regu_reward, 'no_jump': nojump_reward,
                # 'foot_contact': foot_contact_reward, 'body_contact': body_contact_reward,
                'ball_velocity_reward': velocity_reward, 'goal_reward': goal_reward,
                'ball_goal_distance': goal_distance, 'is_success': True if goal_reward > 0 else False
                }
        if self.gui:
            print(info)

        if self.simstep_cnt > 20*self.sim_config.simulation_freq or goal_reward > 0:
            done = True
        else:
            done = False

        return observation, reward, done, info

    def reset(self, seed = None):
        '''set random seed'''
        if seed:
            np.random.seed(seed)
            random.seed(seed)

        self.simstep_cnt = 0

        '''parameters for dynamics'''
        self.friction = self.sim_config.ground_lateral_friction
        self.max_torque = self.robot_config.max_torque
        self.max_velo = self.robot_config.max_torque
        # print('friction factor:',self.friction,'maxtorque:',self.max_torque,'maxvelo',self.max_velo)
        #########################################   dynamics randomization   ##################################
        '''reset ground'''
        p.changeDynamics(bodyUniqueId=self.planeID, linkIndex=-1, lateralFriction=self.friction,
                         rollingFriction=self.sim_config.ground_rolling_friction, spinningFriction=self.sim_config.ground_spinning_friction, restitution=self.sim_config.ground_restitution)

        '''reset ball'''
        p.resetBasePositionAndOrientation(self.ballID, [-1,0,0.07], self.StartOrientation)

        '''reset robot'''
        p.resetBasePositionAndOrientation(self.robotID, self.StartPos, self.StartOrientation)
        for i in range(20):
            p.resetJointState(self.robotID,i,self.rst_qpos[i])
        
        # filter collision
        # p.setCollisionFilterPair(self.robotID, self.robotID, -1, 15, 0)
        # p.setCollisionFilterPair(self.robotID, self.robotID, -1, 9, 0)
        # p.setCollisionFilterPair(self.robotID, self.robotID, 11, 13, 0)
        # p.setCollisionFilterPair(self.robotID, self.robotID, 17, 19, 0)

        '''reset goal. position x:[2, 3), y:[-1, 1)'''
        self.goal_pos = np.random.random(2)
        self.goal_pos[0] += 2
        self.goal_pos[1] *= 2
        self.goal_pos[1] -= 1
        # self.goal_pos = np.array([2, 0])
        p.resetBasePositionAndOrientation(self.goalID, self.goal_pos.tolist()+[0], self.StartOrientation)

        '''compute initial observation'''
        pos, vel, omega, grav, qpos, qvel, react, torq, ball_pos, ball_velocity, goal_pos = self.get_state()
        self.last_action = np.array(qpos)
        ball_pos_relative, goal_pos_relative = self.compute_relative_pos(pos, ball_pos, goal_pos)
        observation = qpos + omega + grav + vel[:2] + goal_pos_relative
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        return observation
