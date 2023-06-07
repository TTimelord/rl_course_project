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

class KickBallImitation(KickBall):
    def __init__(self, connect_GUI=False):
        super(KickBallImitation, self).__init__(connect_GUI)
        self.back_swing = False  # leg swing back flag
        self.back_swing_leg_is_left = None
        self.touch_ball_with_correct_foot = False
        p.resetDebugVisualizerCamera(cameraDistance=2, cameraYaw=0, cameraPitch=-30,
                            cameraTargetPosition=[1.0,-0.0,0.1])
    
    def step(self, action):
        '''filter action and step simulation'''
        new_action = np.array(action) + np.array(self.rst_qpos)
        real_action = self.robot_config.lpf_ratio * new_action + (1 - self.robot_config.lpf_ratio) * self.last_action
        self.last_action = real_action
        self.set_qpos_all(real_action)
        collision_buffer=[]
        for s in range(self.sim_per_control):
            p.stepSimulation()
            collision_buffer += p.getContactPoints(bodyA=self.robotID, bodyB=self.ballID) + \
                                p.getContactPoints(bodyA=self.robotID, bodyB=self.ballID)
            self.simstep_cnt += 1
            if self.gui:
                time.sleep(1/600.)  # slow down

        '''compute observation'''
        pos, vel, omega, grav, qpos, qvel, react, torq, ball_pos, ball_vel, goal_pos = self.get_state()
        ball_pos_relative, goal_pos_relative = self.compute_relative_pos(pos, ball_pos, goal_pos)
        observation = qpos + self.qpos_buffer + omega + grav + ball_pos_relative + ball_vel[:2] + goal_pos_relative
        self.qpos_buffer = qpos

        '''compute reward'''
        # ori_reward = 1.0 * rbf_reward(grav, [0, 0, -1], -1.)
        height = pos[2]
        # height_reward = 1.0 * rbf_reward(height, self.robot_config.center_height * 0.999, -10.)
        # print(height)
        # if height >=0.3:
        #     height_reward = 3*np.log((height-0.29999)/0.1)
        # else:
        #     height_reward = 3*np.log((0.3-0.29999)/0.1)

        '''back swing'''
        back_swing_reward = 0

        # compute if body is upright
        upright = False
        if np.dot(grav, [0, 0, -1]) > np.cos(pi/6):
            upright = True
        
        # left_foot_pos = p.getLinkState(self.robotID, 5)[0]
        # right_foot_pos = p.getLinkState(self.robotID, 19)[0]
        left_foot_pos,_,_,_,_,_,left_foot_vel,_ = p.getLinkState(self.robotID, 5, computeLinkVelocity=1)
        right_foot_pos,_,_,_,_,_,right_foot_vel,_ = p.getLinkState(self.robotID, 19, computeLinkVelocity=1)

        back_swing_x_thresh = -0.02
        back_swing_z_thresh = 0.04

        left_foot_back_swing = left_foot_pos[0] - pos[0] < back_swing_x_thresh and left_foot_pos[2] > back_swing_z_thresh
        right_foot_back_swing = right_foot_pos[0] - pos[0] < back_swing_x_thresh and right_foot_pos[2] > back_swing_z_thresh

        if not self.back_swing and (left_foot_back_swing or right_foot_back_swing) and upright:  # the first time back swing happens when body is upright
            back_swing_reward = 50
            self.back_swing = True
            if left_foot_back_swing:
                self.back_swing_leg_is_left = True
            elif right_foot_back_swing:
                self.back_swing_leg_is_left = False

        foot_velocity_reward = 0
        ball_velocity_reward = 0
        goal_reward = 0
        touch_reward = 0  # given once when the robot touch the ball
        if not self.touch_ball_with_correct_foot and self.back_swing:
            if self.back_swing_leg_is_left:
                foot_velocity_reward = 5*left_foot_vel[0]
            else:
                foot_velocity_reward = 5*right_foot_vel[0]
            if collision_buffer:
                for point in collision_buffer:
                    if self.back_swing_leg_is_left:
                        if point[3]==5:
                            self.touch_ball_with_correct_foot=True
                            touch_reward = 50
                    else:
                        if point[3]==19:
                            self.touch_ball_with_correct_foot=True
                            touch_reward = 50
        collision_buffer = []
        
        if self.touch_ball_with_correct_foot:
            ball_velocity_reward = self.compute_relative_velocity(ball_vel, ball_pos, goal_pos) * 2

            ball_goal_distance = np.linalg.norm(np.array(ball_pos)[:2] - np.array(goal_pos), ord=2)
            if ball_goal_distance < self.sim_config.goal_radius:
                self.hit_target = True
                goal_reward = 1000 
            
        termination = height < 0.33 #0.3

        '''total reward'''
        reward = back_swing_reward + foot_velocity_reward + touch_reward + ball_velocity_reward + goal_reward + termination*-100
        info = {
                # 'ori_reward': ori_reward, 'height_reward': height_reward,
                # 'omega_reward': omega_reward, 'torq_regu': joint_torq_regu_reward,
                # 'velo_regu': joint_velo_regu_reward, 'no_jump': nojump_reward,
                # 'foot_contact': foot_contact_reward, 'body_contact': body_contact_reward,
                'back_swing':self.back_swing,'touch_ball':self.touch_ball_with_correct_foot,
                'total_reward': reward, 'foot_velocity_reward':foot_velocity_reward, 'touch_reward': touch_reward,
                'ball_velocity_reward': ball_velocity_reward,
                'is_success': True if self.hit_target else False,
                }
        if self.gui:
            print(info)

        if self.simstep_cnt > 3*self.sim_config.simulation_freq or self.hit_target or termination:  # or termination:  # comment this when training kick_new
            done = True
        else:
            done = False
            
        # if termination:
        #     done = True

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
        self.max_velo = self.robot_config.max_velocity
        # print('friction factor:',self.friction,'maxtorque:',self.max_torque,'maxvelo',self.max_velo)
        #########################################   dynamics randomization   ##################################
        '''reset ground'''
        p.changeDynamics(bodyUniqueId=self.planeID, linkIndex=-1, lateralFriction=self.friction,
                         rollingFriction=self.sim_config.ground_rolling_friction, spinningFriction=self.sim_config.ground_spinning_friction, restitution=self.sim_config.ground_restitution)

        '''reset ball'''
        p.resetBasePositionAndOrientation(self.ballID, [0.3,0,0.07], self.StartOrientation)

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
        pos, vel, omega, grav, qpos, qvel, react, torq, ball_pos, ball_vel, goal_pos = self.get_state()
        self.last_action = np.array(qpos)
        self.qpos_buffer = qpos
        ball_pos_relative, goal_pos_relative = self.compute_relative_pos(pos, ball_pos, goal_pos)
        observation = qpos + self.qpos_buffer + omega + grav + ball_pos_relative + ball_vel[:2] + goal_pos_relative
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        self.hit_target = False
        self.back_swing = False
        self.back_swing_leg_is_left = None
        self.touch_ball_with_correct_foot = False
        return observation
