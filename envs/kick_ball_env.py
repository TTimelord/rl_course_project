import gym
from gym import error, spaces, utils
import pybullet as p
import pybullet_data
from math import pi
import numpy as np
from scipy.spatial.transform import Rotation
import time
from config import RobotConfig, SimulationConfig

def get_omega_imu(q, omega):
    rot_ = Rotation.from_quat(q)
    mat_ = rot_.as_matrix()
    vec_ = np.dot(np.transpose(mat_), np.array([[x] for x in omega]))
    return np.reshape(vec_,(-1,))

def get_gravity_vec(q):
    rot_ = Rotation.from_quat(q)
    mat_ = rot_.as_matrix()
    vec_ = np.dot(np.transpose(mat_), np.array([[0.], [0.], [-1.]]))
    out_ = np.reshape(vec_, (-1,))
    return out_

def rbf_reward(x,xhat,alpha):
    x = np.array(x)
    xhat = np.array(xhat)
    return np.exp(alpha * np.sum(np.square(x-xhat)))

class KickBall(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, connect_GUI=False):
        self.simstep_cnt = 0
        self.gui = connect_GUI
        self.robot_config = RobotConfig()
        self.sim_config = SimulationConfig()
        self.sim_per_control = self.sim_config.simulation_freq // self.robot_config.control_freq
        self.mos_height = self.robot_config.center_height
        if connect_GUI:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetDebugVisualizerCamera(cameraDistance=1, cameraYaw=70, cameraPitch=-40,
                                 cameraTargetPosition=[0,0,0.2])
        lower_joints = self.robot_config.lower_joints
        upper_joints = self.robot_config.upper_joints
        self.action_space = spaces.Box(low=np.array(lower_joints),high=np.array(upper_joints))
        self.observation_space = spaces.Box(low=np.array(lower_joints  + [-10]*3+[-1]*3),  # joints, omega, grav
                                            high=np.array(upper_joints + [10]*3 + [1]*3))

        self.initial_configuration = [
            ([0, 0, 0.435521], [0] * 2 + [0] + [0] * 2 + [0] + [0] * 14, [0, 0, 0]),  # standing
        ]

    def step(self, action):
        new_action = np.array(action)
        real_action = self.robot_config.lpf_ratio * new_action + (1 - self.robot_config.lpf_ratio) * self.last_action
        self.last_action = real_action
        self.set_qpos_all(real_action)
        for s in range(self.sim_per_control):
            p.stepSimulation()
            self.simstep_cnt += 1
            if self.gui:
                time.sleep(1/600.)  # slow down

        qpos, qvel, omega, grav, react, torq, ball_vel = self.get_state()
        observation = qpos + omega + grav

        ori_reward = 0.333 * rbf_reward(grav, [0, 0, -1], -5)
        height = p.getBasePositionAndOrientation(self.robotID)[0][2]
        height_reward = 0.667 * rbf_reward(height, self.robot_config.center_height * 0.999, -60.)
        omega_reward = 0.067 * rbf_reward(omega, [0, 0, 0], -0.05)
        joint_torq_regu_reward = 0.067 * rbf_reward(torq, 0, -0.5)
        joint_velo_regu_reward = 0.067 * rbf_reward(qvel, 0, -0.05)
        foot_contact, body_contact = self.contact_reward()
        foot_contact_reward = 0.033 * foot_contact
        body_contact_reward = 0.067 * body_contact
        dists = [c[8] for c in p.getClosestPoints(self.robotID, self.planeId, 1.)]
        dist = min(dists + [1.])
        nojump_reward = 0.033 * rbf_reward(dist, 0, -100.0)
        ball_velocity_reward = np.linalg.norm(ball_vel, ord=2)
        # reward = ori_reward + height_reward + omega_reward + \
        #          joint_torq_regu_reward + joint_velo_regu_reward + \
        #          foot_contact_reward + body_contact_reward + nojump_reward
        reward = ball_velocity_reward
        info = {'ori_reward': ori_reward, 'height_reward': height_reward,
                'omega_reward': omega_reward, 'torq_regu': joint_torq_regu_reward,
                'velo_regu': joint_velo_regu_reward, 'no_jump': nojump_reward,
                'foot_contact': foot_contact_reward, 'body_contact': body_contact_reward,
                'ball_velocity_reward': ball_velocity_reward
                }
        if self.gui:
            # print(info)
            pass

        done = False
        if self.simstep_cnt > 300:
            done = True

        return observation, reward, done, info

    def reset(self, seed=0):
        np.random.seed(seed)
        p.resetSimulation()
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        if self.gui:
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.setGravity(0, 0, -9.81)
        p.setPhysicsEngineParameter(fixedTimeStep= 1./self.sim_config.simulation_freq)

        self.simstep_cnt = 0

        # self.friction = random.uniform(0.5, 1.4)
        # self.maxtorque = random.uniform(4.8, 7.0)
        # self.maxvelo = random.uniform(4.5, 7.0)
        self.friction = self.sim_config.ground_lateral_friction
        self.max_torque = self.robot_config.max_torque
        self.max_velo = self.robot_config.max_torque
        print('friction factor:',self.friction,'maxtorque:',self.max_torque,'maxvelo',self.max_velo)
        #########################################   dynamics randomization   ##################################
        # load ground
        planeID = p.loadURDF("plane.urdf", useFixedBase=True)
        self.planeId = planeID
        p.changeDynamics(bodyUniqueId=planeID, linkIndex=-1, lateralFriction=self.friction,
                         rollingFriction=self.sim_config.ground_rolling_friction, spinningFriction=self.sim_config.ground_spinning_friction, restitution=self.sim_config.ground_restitution)

        self.ballID = p.loadURDF("soccerball.urdf",[0.3,0,0.07], globalScaling=0.14)
        p.changeDynamics(self.ballID,-1, mass=0.25, linearDamping=0, angularDamping=0, rollingFriction=0.001, spinningFriction=0.001, restitution=0.5)
        p.changeVisualShape(self.ballID,-1,rgbaColor=[0.8,0.8,0.8,1])

        self.StartPos, rst_qpos,rpy_ini = self.initial_configuration[0]
        self.StartOrientation = p.getQuaternionFromEuler(rpy_ini)
        self.robotID = p.loadURDF(self.robot_config.urdf_path, [0,0,self.robot_config.center_height+0.5], self.StartOrientation,
                                flags = p.URDF_MERGE_FIXED_LINKS|p.URDF_USE_SELF_COLLISION|p.URDF_USE_INERTIA_FROM_FILE|p.URDF_MAINTAIN_LINK_ORDER)
        p.resetBasePositionAndOrientation(self.robotID, self.StartPos, self.StartOrientation)
        # p.setCollisionFilterPair(self.robotID, self.robotID, -1, 15, 0)
        # p.setCollisionFilterPair(self.robotID, self.robotID, -1, 9, 0)
        # p.setCollisionFilterPair(self.robotID, self.robotID, 11, 13, 0)
        # p.setCollisionFilterPair(self.robotID, self.robotID, 17, 19, 0)
        for i in range(20):
            p.resetJointState(self.robotID,i,rst_qpos[i])

        qpos, velo, omega, grav, react, torq, ball_velo = self.get_state()
        self.last_action = np.array(qpos)
        observation = qpos + omega + grav
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        return observation

    def render(self, mode='human'):
        pass
        return

    def close(self):
        p.disconnect()

    def get_reward_dict(self):
        # TODO: move reward computing here
        pass

    def contact_reward(self):
        """return foot,body
        foot = 1 if in contact with ground, otherwise 0
        body = 0 if in contact with ground, otherwise 1
        """
        clp = p.getClosestPoints(self.robotID, self.planeId, 1e-4)
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
        # robot
        states = p.getJointStates(self.robotID, [x for x in range(20)])
        qpos = [s[0] for s in states]
        qvel = [s[1] for s in states]
        react = [s[2] for s in states]
        torq = [s[3] for s in states]
        omega = list(p.getBaseVelocity(self.robotID)[1])  # base omega
        q = p.getBasePositionAndOrientation(self.robotID)[1]
        grav = get_gravity_vec(q).tolist()
        omega = get_omega_imu(q,omega).tolist()  # imu omega

        # ball
        ball_velocity = p.getBaseVelocity(self.ballID)[0]
        
        return qpos, qvel, omega, grav, react, torq, ball_velocity
