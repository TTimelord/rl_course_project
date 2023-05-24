import gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from config import RobotConfig
import envs
import time

robot_config = RobotConfig()
eval_env = gym.make('KickBall-v0', connect_GUI=True)

# model = PPO("MlpPolicy", eval_env, verbose=1, tensorboard_log="./logs/")

# load trained agent
model = PPO.load('models/PPO_random_goal.zip', print_system_info=True)

# evaluation with GUI
# evaluate_policy(model, eval_env, n_eval_episodes=10)
while True:
    obs = eval_env.reset()
    time.sleep(1.)
    for i in range(3 * robot_config.control_freq):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = eval_env.step(action)
        time.sleep(1./robot_config.control_freq)
        if done:
            break
        print(reward)