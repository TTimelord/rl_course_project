import gym
from gym import error, spaces, utils
import pybullet as p
import pybullet_data
from math import pi
import numpy as np
import time
import random
from config import RobotConfig, SimulationConfig
from utils import get_gravity_vec, get_omega_imu, rbf_reward

class KickBall(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, connect_GUI=False):
        self.simstep_cnt = 0
        self.gui = connect_GUI
        self.robot_config = RobotConfig()
        self.sim_config = SimulationConfig()
        self.sim_per_control = self.sim_config.simulation_freq // self.robot_config.control_freq
        self.mos_height = self.robot_config.center_height
        self.hit_target = False
        if connect_GUI:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetDebugVisualizerCamera(cameraDistance=2, cameraYaw=70, cameraPitch=-60,
                                 cameraTargetPosition=[2,-0.5,0.2])
        lower_joints = self.robot_config.lower_joints
        upper_joints = self.robot_config.upper_joints
        self.action_space = spaces.Box(low=np.array(lower_joints), high=np.array(upper_joints))
        self.observation_space = spaces.Box(low=np.array(lower_joints  + [-10]*3+[-1]*3 + [-3]*6),  # joints, omega, grav, ball_pos, ball_vel, goal_pos
                                            high=np.array(upper_joints + [10]*3 + [1]*3 + [3]*6))

        self.initial_configuration = [
            ([0, 0, 0.435521], [0] * 2 + [0] + [0] * 2 + [0] + [0] * 14, [0, 0, 0]),  # standing
        ]

        '''initialize simulation'''
        p.resetSimulation()
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        if self.gui:
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.setGravity(0, 0, -9.81)
        p.setPhysicsEngineParameter(fixedTimeStep= 1./self.sim_config.simulation_freq)

        '''load plane'''
        self.planeID = p.loadURDF("plane.urdf", useFixedBase=True)
        
        '''load football'''
        self.ballID = p.loadURDF("soccerball.urdf", [1, 0, 0.5], globalScaling=0.14)
        p.changeDynamics(self.ballID,-1, mass=0.25, linearDamping=0, angularDamping=0, rollingFriction=0.001, spinningFriction=0.001, restitution=0.5)
        p.changeVisualShape(self.ballID,-1,rgbaColor=[0.8,0.8,0.8,1])

        '''load robot'''
        self.robotID = p.loadURDF(self.robot_config.urdf_path, [0,0,self.robot_config.center_height+0.5],
                                flags = p.URDF_MERGE_FIXED_LINKS|p.URDF_USE_SELF_COLLISION|p.URDF_USE_INERTIA_FROM_FILE|p.URDF_MAINTAIN_LINK_ORDER)
        self.StartPos, self.rst_qpos,rpy_ini = self.initial_configuration[0]
        self.StartOrientation = p.getQuaternionFromEuler(rpy_ini)

        '''load goal'''
        goalShapeID = p.createVisualShape(p.GEOM_CYLINDER, radius=self.sim_config.goal_radius, length=0.02, rgbaColor=[0.1,0.9,0.1,0.7])
        self.goalID = p.createMultiBody(baseVisualShapeIndex=goalShapeID, basePosition=[2, 0, 0])  # for visualization

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
        pos, vel, omega, grav, qpos, qvel, react, torq, ball_pos, ball_vel, goal_pos = self.get_state()
        ball_pos_relative, goal_pos_relative = self.compute_relative_pos(pos, ball_pos, goal_pos)
        observation = qpos + omega + grav + ball_pos_relative + ball_vel[:2] + goal_pos_relative

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
        ball_velocity_reward = self.compute_relative_velocity(ball_vel, ball_pos, goal_pos)
        ball_goal_distance = np.linalg.norm(np.array(ball_pos)[:2] - np.array(goal_pos), ord=2)
        goal_reward = 0
        if ball_goal_distance < self.sim_config.goal_radius:
            self.hit_target = True
            goal_reward = 1000 

        # reward = ori_reward + height_reward + omega_reward + \
        #          joint_torq_regu_reward + joint_velo_regu_reward + \
        #          foot_contact_reward + body_contact_reward + nojump_reward

        reward = ball_velocity_reward + goal_reward
        info = {
                'ori_reward': ori_reward, 'height_reward': height_reward,
                # 'omega_reward': omega_reward, 'torq_regu': joint_torq_regu_reward,
                # 'velo_regu': joint_velo_regu_reward, 'no_jump': nojump_reward,
                # 'foot_contact': foot_contact_reward, 'body_contact': body_contact_reward,
                'ball_velocity_reward': ball_velocity_reward, 'goal_reward': goal_reward,
                'ball_goal_distance': ball_goal_distance, 'is_success': True if goal_reward > 0 else False
                }
        if self.gui:
            print(info)

        if self.simstep_cnt > 5*self.sim_config.simulation_freq or self.hit_target:
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
        p.resetBasePositionAndOrientation(self.ballID, [0.2,0,0.07], self.StartOrientation)

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
        ball_pos_relative, goal_pos_relative = self.compute_relative_pos(pos, ball_pos, goal_pos)
        observation = qpos + omega + grav + ball_pos_relative + ball_vel[:2] + goal_pos_relative
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        self.hit_target = False
        return observation

    def render(self, mode='human'):
        pass
        return

    def close(self):
        p.disconnect()

    def contact_reward(self):
        """return foot,body
        foot = 1 if in contact with ground, otherwise 0
        body = 0 if in contact with ground, otherwise 1
        """
        clp = p.getClosestPoints(self.robotID, self.planeID, 1e-4)
        links = [c[3] for c in clp]
        # feet 1357, back -1
        foot = 0 # foot - ground
        if (5 in links):
            foot += 0.5
        if (5 in links):
            foot += 0.5

        body = 1  # body - ground
        for l in links:
            if l!=5 and l!=19:
                body = 0
        return foot,body

    def set_qpos_all(self, arr):  # positive = forward
        for i in range(20):
            self.set_qpos(i,arr[i])

    def set_qpos(self, jid, pos):
        p.setJointMotorControl2(self.robotID, jid, controlMode=p.POSITION_CONTROL, 
                                targetPosition=pos, force=self.max_torque, maxVelocity=self.max_velo)

    def get_state(self):
        '''robot'''
        pos, q = p.getBasePositionAndOrientation(self.robotID)
        vel = list(p.getBaseVelocity(self.robotID)[0])
        omega = list(p.getBaseVelocity(self.robotID)[1])  # base omega
        grav = get_gravity_vec(q).tolist()
        omega = get_omega_imu(q,omega).tolist()  # imu omega
        states = p.getJointStates(self.robotID, [x for x in range(20)])
        qpos = [s[0] for s in states]
        qvel = [s[1] for s in states]
        react = [s[2] for s in states]
        torq = [s[3] for s in states]

        '''ball'''
        ball_pos = list(p.getBasePositionAndOrientation(self.ballID)[0])
        ball_velocity = list(p.getBaseVelocity(self.ballID)[0])

        '''goal'''
        goal_pos = self.goal_pos.tolist()
        
        return pos, vel, omega, grav, qpos, qvel, react, torq, ball_pos, ball_velocity, goal_pos
    
    def compute_relative_pos(self, pos, ball_pos, goal_pos):
        """
        compute relative position in xy plane
        """
        pos = np.array(pos)[:2]
        ball_pos = np.array(ball_pos)[:2]
        goal_pos = np.array(goal_pos)[:2]
        ball_pos_relative = ball_pos - pos
        goal_pos_relative = goal_pos - pos
        return ball_pos_relative.tolist(), goal_pos_relative.tolist()
    
    def compute_relative_velocity(self, vel, pos, goal_pos):
        """
        compute relative velocity towards goal
        """
        vel = np.array(vel)[:2]
        pos = np.array(pos)[:2]
        goal_pos = np.array(goal_pos)[:2]
        relative_pos = goal_pos - pos
        vel_relative = np.dot(vel, relative_pos)/np.linalg.norm(relative_pos)
        return vel_relative
