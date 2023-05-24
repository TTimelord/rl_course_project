import pybullet as p
import time
import pybullet_data
from math import pi
import numpy as np
from scipy.spatial.transform import Rotation
from config import RobotConfig, SimulationConfig

def get_gravity_vec(q):
    rot_ = Rotation.from_quat(q)
    mat_ = rot_.as_matrix()
    vec_ = np.dot(np.transpose(mat_), np.array([[0.], [0.], [-1.]]))
    return vec_

def reset_sim():
    p.resetSimulation()
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
    p.resetDebugVisualizerCamera(cameraDistance=1, cameraYaw=70, cameraPitch=-40,
                                 cameraTargetPosition=[0,0,0.2])
    p.setGravity(0, 0, -9.81)
    p.setPhysicsEngineParameter(fixedTimeStep=1./sim_config.simulation_freq)
    planeId = p.loadURDF("plane.urdf",useFixedBase=True)
    p.changeDynamics(bodyUniqueId=planeId,linkIndex=-1,lateralFriction=sim_config.ground_lateral_friction,
                     rollingFriction=sim_config.ground_rolling_friction,spinningFriction=sim_config.ground_spinning_friction,linearDamping=0,angularDamping=0,
                     mass=0,localInertiaDiagonal=[0, 0, 0],restitution=sim_config.ground_restitution)
    return planeId

def set_pos(uid,jid,pos):
    p.setJointMotorControl2(uid,jid,force=5.0,
                                controlMode= p.POSITION_CONTROL,targetPosition = pos,maxVelocity=4)
def get_state(uid):
    states = p.getJointStates(uid,[0,1,2,3,4,5,6,7])
    pos = [s[0] for s in states]
    velo = [s[1] for s in states]
    react = [s[2] for s in states]
    torq = [s[3] for s in states]
    omega = p.getBaseVelocity(uid)[1]
    q = p.getBasePositionAndOrientation(uid)[1]
    grav = np.reshape(get_gravity_vec(q),(-1,))
    return pos,velo,list(omega),grav.tolist(),react,torq


sim_config = SimulationConfig()
robot_config = RobotConfig()

physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally

planeId = reset_sim()
mos_height = 0.435521
# StartPos,q = ([0, 0, mos_height], [0, 0, 0]) # stand
StartPos, rst_pos, q = ([0, 0, 0.435521], [0] * 2 + [0] + [0] * 2 + [0] + [0] * 14, [0, 0, 0])
StartOrientation = p.getQuaternionFromEuler(q)
robotID = p.loadURDF(robot_config.urdf_path, [0,0,mos_height+0.5], StartOrientation,
                    flags = p.URDF_MERGE_FIXED_LINKS|p.URDF_USE_SELF_COLLISION|p.URDF_USE_INERTIA_FROM_FILE)
p.resetBasePositionAndOrientation(robotID, StartPos, StartOrientation)
# p.setCollisionFilterPair(mosid,mosid,-1,15,0)
# p.setCollisionFilterPair(mosid, mosid, -1, 9, 0)
# p.setCollisionFilterPair(mosid, mosid, 11, 13, 0)
# p.setCollisionFilterPair(mosid, mosid, 17, 19, 0)
lower_joints = robot_config.lower_joints
upper_joints = robot_config.upper_joints
n_joints = p.getNumJoints(robotID)
# rst_pos = [0 for i in range(n_joints)]
joint_name = []
# rst_pos[3]= pi/2
# rst_pos[6]=-pi/2

for i in range(n_joints):
    joint_info = p.getJointInfo(robotID, i)
    print(joint_info[0], joint_info[1],joint_info[2], joint_info[8:13])
    joint_name.append(str(joint_info[1]))

para_ids = []
for i in range(n_joints):
    p.resetJointState(robotID, i, rst_pos[i])
    id_ = p.addUserDebugParameter(joint_name[i],lower_joints[i]-1e-8,upper_joints[i]+1e-8,rst_pos[i])
    para_ids.append(id_)

ballID = p.loadURDF("soccerball.urdf",[0.5,0,1], globalScaling=0.14)
p.changeDynamics(ballID,-1, mass=0.25, linearDamping=0, angularDamping=0, rollingFriction=0.001, spinningFriction=0.001, restitution=0.5)
p.changeVisualShape(ballID,-1,rgbaColor=[0.8,0.8,0.8,1])

# goal
goal_pos = np.random.random(2)
goal_pos[0] += 2
goal_pos[1] *= 2
goal_pos[1] -= 1

goalShapeID = p.createVisualShape(p.GEOM_CYLINDER, radius=sim_config.goal_radius, length=0.02, rgbaColor=[0.1,0.9,0.1,0.7])
goalID = p.createMultiBody(baseVisualShapeIndex=goalShapeID, basePosition=goal_pos.tolist()+[0])

# test in real time
p.setRealTimeSimulation(1)
for i in range(3600 * sim_config.simulation_freq):
    # p.stepSimulation()
    time.sleep(1./sim_config.simulation_freq)
    for j in range(n_joints):
        set_pos(robotID,j,p.readUserDebugParameter(para_ids[j]))
    cubePos, cubeOrn = p.getBasePositionAndOrientation(robotID)
    print(cubePos, p.getEulerFromQuaternion(cubeOrn))

p.disconnect()
