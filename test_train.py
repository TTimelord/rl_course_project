import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from config import RobotConfig
# import envs
import time
import torch
from utils import AutoSaveCallback
from envs.kick_ball_env_wq import KickBall57

robot_config = RobotConfig()


# evaluation with GUI
eval_env = gym.make('KickBall57-v0', connect_GUI=True)
while True:
    obs = eval_env.reset()
    while True:
        # action = [0,0,0.8,-0.8,0,0,0,0,0,0,
        #           0,1,2,-1,-2,0,0,0,0,0]
        action = [0]*20
        obs, reward, done, info = eval_env.step(action)
        if done:
            break
        time.sleep(1./robot_config.control_freq)
