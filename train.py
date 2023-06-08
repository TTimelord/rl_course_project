import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from config import RobotConfig
import envs
import time
import torch
from stable_baselines3.common.callbacks import CheckpointCallback

robot_config = RobotConfig()

'''train parameters'''
gym_id = 'KickBallImitation-v0'  # change the env you want to train here
n_procs = 24
train_env = make_vec_env(gym_id, n_envs=n_procs, vec_env_cls=SubprocVecEnv, vec_env_kwargs=dict(start_method='fork'))
policy_kwargs = dict(activation_fn=torch.nn.ReLU,
                     net_arch=dict(pi=[512, 512], vf=[512, 512]))
model = PPO("MlpPolicy", train_env, verbose=1, tensorboard_log="./logs/", learning_rate=1e-4, 
            n_steps=1024, gae_lambda=0.8, gamma=0.99, batch_size=128, n_epochs=5, policy_kwargs=policy_kwargs)

'''load the previous model if you want to continue training'''
# model = PPO.load('models/PPO_imitation.zip', print_system_info=True, env=train_env)

print(model.policy)
checkpoint_callback = CheckpointCallback(save_freq=1e6//n_procs, save_path='./models/auto_save/', name_prefix=gym_id)

'''training (set reset_num_timesteps=False if continue training)'''
model.learn(total_timesteps=1e7, tb_log_name=gym_id, reset_num_timesteps=True, callback=checkpoint_callback)

'''save trained agent'''
model.save('./models/' + gym_id)

'''evaluation with GUI'''
eval_env = gym.make(gym_id, connect_GUI=True)
while True:
    obs = eval_env.reset()
    while True:
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = eval_env.step(action)
        if done:
            break
        time.sleep(1./robot_config.control_freq)