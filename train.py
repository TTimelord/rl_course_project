import gym
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from config import RobotConfig
import envs
import time

robot_config = RobotConfig()

# train the agent using multiprocessing
n_procs = 10
train_env = make_vec_env('KickBall-v0', n_envs=n_procs, vec_env_cls=SubprocVecEnv, vec_env_kwargs=dict(start_method='fork'))
model = A2C("MlpPolicy", train_env, verbose=1, tensorboard_log="./logs/")
model.learn(total_timesteps=1000, tb_log_name="first_run")

# save trained agent
model.save('./models/A2C_first_run.zip')

# evaluation with GUI
eval_env = gym.make('KickBall-v0', connect_GUI=True)
while True:
    obs = eval_env.reset()
    for i in range(5 * robot_config.control_freq):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = eval_env.step(action)
        eval_env.render("human")
        time.sleep(1./robot_config.control_freq)