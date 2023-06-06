import gym
from gym import error, spaces, utils
import pybullet as p
import pybullet_data
from math import pi
import numpy as np
import time
import random
from config import RobotConfig, SimulationConfig
from envs.kick_ball_env import KickBall
from utils import get_gravity_vec, get_omega_imu, rbf_reward

class KickBallStand(KickBall):
    def __init__(self, connect_GUI=False):
        super(KickBallStand, self).__init__(connect_GUI)
    
    def step(self, action):
        '''filter action and step simulation'''
        new_action = np.array(action) + np.array(self.rst_qpos)
        real_action = self.robot_config.lpf_ratio * new_action + (1 - self.robot_config.lpf_ratio) * self.last_action
        self.last_action = real_action
        self.set_qpos_all(real_action)
        for s in range(self.sim_per_control):
            p.stepSimulation()
            self.simstep_cnt += 1
            if self.gui:
                time.sleep(1/600.)  # slow down

        '''compute observation'''
        pos, vel, omega, grav, qpos, qvel, react, torq, ball_pos, ball_vel, goal_pos = self.get_state()
        ball_pos_relative, goal_pos_relative = self.compute_relative_pos(pos, ball_pos, goal_pos)
        observation = qpos + omega + grav + ball_pos_relative + ball_vel[:2] + goal_pos_relative

        '''compute reward'''
        ori_reward = 1.0 * rbf_reward(grav, [0, 0, -1], -1.)
        height = pos[2]
        height_reward = 1.0 * rbf_reward(height, self.robot_config.center_height * 0.999, -10.)
        # print(height)
        if height >=0.3:
            height_reward = 3*np.log((height-0.29999)/0.1)
        else:
            height_reward = 3*np.log((0.3-0.29999)/0.1)
        # omega_reward = 0.067 * rbf_reward(omega, [0, 0, 0], -0.05)
        # joint_torq_regu_reward = 0.067 * rbf_reward(torq, 0, -0.5)
        # joint_velo_regu_reward = 0.067 * rbf_reward(qvel, 0, -0.05)
        # foot_contact, body_contact = self.contact_reward()
        # foot_contact_reward = 0.033 * foot_contact
        # body_contact_reward = 0.067 * body_contact
        # dists = [c[8] for c in p.getClosestPoints(self.robotID, self.planeId, 1.)]
        # dist = min(dists + [1.])
        # nojump_reward = 0.033 * rbf_reward(dist, 0, -100.0)
        ball_velocity_reward = self.compute_relative_velocity(ball_vel, ball_pos, goal_pos)
        ball_goal_distance = np.linalg.norm(np.array(ball_pos)[:2] - np.array(goal_pos), ord=2)
        goal_reward = 0
        if ball_goal_distance < self.sim_config.goal_radius:
            self.hit_target = True
            goal_reward = 1000 
            
        termination = height<0.33#0.3

        # reward = ori_reward + height_reward + omega_reward + \
        #          joint_torq_regu_reward + joint_velo_regu_reward + \
        #          foot_contact_reward + body_contact_reward + nojump_reward

        reward = ball_velocity_reward + goal_reward +height_reward*0+ termination*-10#-10
        info = {
                'ori_reward': ori_reward, 'height_reward': height_reward,
                # 'omega_reward': omega_reward, 'torq_regu': joint_torq_regu_reward,
                # 'velo_regu': joint_velo_regu_reward, 'no_jump': nojump_reward,
                # 'foot_contact': foot_contact_reward, 'body_contact': body_contact_reward,
                'ball_velocity_reward': ball_velocity_reward, 'goal_reward': goal_reward,
                'ball_goal_distance': ball_goal_distance, 'is_success': True if goal_reward > 0 else False,
                'height':height, 'is_dead':termination
                }
        if self.gui:
            print(info)

        # if self.simstep_cnt > 5*self.sim_config.simulation_freq or goal_reward > 0:  # comment this when training kick_new
        #     done = True
        # else:
        done = False
            
        if termination:
            done = True

        return observation, reward, done, info
