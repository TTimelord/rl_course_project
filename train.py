import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from config import RobotConfig
import envs
import time
import torch
from utils import AutoSaveCallback

robot_config = RobotConfig()

# train the agent using multiprocessing
n_procs = 10
train_env = make_vec_env('KickBall-v0', n_envs=n_procs, vec_env_cls=SubprocVecEnv, vec_env_kwargs=dict(start_method='fork'))
policy_kwargs = dict(activation_fn=torch.nn.ReLU,
                     net_arch=dict(pi=[512, 512], vf=[512, 512]))
model = PPO("MlpPolicy", train_env, verbose=1, tensorboard_log="./logs/", learning_rate=1e-4, 
            gae_lambda=0.8, gamma=0.99, batch_size=32, n_epochs=5, policy_kwargs=policy_kwargs)
print(model.policy)
autosave_callback = AutoSaveCallback(1e5, './models')
model.learn(total_timesteps=1e6, tb_log_name="test_run", callback=autosave_callback)

# save trained agent
model.save('./models/PPO_random_goal.zip')

# evaluation with GUI
eval_env = gym.make('KickBall-v0', connect_GUI=True)
while True:
    obs = eval_env.reset()
    while True:
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = eval_env.step(action)
        if done:
            break
        time.sleep(1./robot_config.control_freq)