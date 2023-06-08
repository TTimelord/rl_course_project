import gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from config import RobotConfig
import envs
import time

robot_config = RobotConfig()
eval_env = gym.make('KickBallImitation-v0', connect_GUI=True)

'''load trained agent'''
model = PPO.load('models/KickBallImitation-v0.zip', print_system_info=True)

'''evaluation with GUI'''
# evaluate_policy(model, eval_env, n_eval_episodes=10)
while True:
    obs = eval_env.reset()
    time.sleep(0.5)
    while True:
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = eval_env.step(action)
        time.sleep(1./robot_config.control_freq)
        if done:
            break