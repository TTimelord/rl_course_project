import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from config import RobotConfig
import envs
import time

robot_config = RobotConfig()
eval_env = gym.make('KickBall-v0', connect_GUI=True)

# model = PPO("MlpPolicy", eval_env, verbose=1, tensorboard_log="./logs/")

# load trained agent
model = PPO.load('./models/PPO_test_run.zip')

# evaluation with GUI
while True:
    obs = eval_env.reset()
    for i in range(5 * robot_config.control_freq):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = eval_env.step(action)
        eval_env.render("human")
        time.sleep(1./robot_config.control_freq)
        print(reward)