# Installation
1. clone this repo
2. install pytorch

    https://pytorch.org/get-started/locally/

3. install Stable-Baselines3
```
pip install stable-baselines3[extra]
```
(version 1.8.0)
Documentation: https://stable-baselines3.readthedocs.io/

Tutorial: https://github.com/araffin/rl-tutorial-jnrr19/blob/sb3

4. install pybullet
```
pip install pybullet
```
Documentation: https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/

# Code Structure
- rl_course_project
    - `assets/` \
        store URDF file of the robot
    - `envs/` \
        存放强化学习 gym 环境(目前使用的stable-baselines3是1.8.0版本，因此还使用的是gym而不是gymnasium(https://gymnasium.farama.org/))
        - `kick_ball_env.py`
        踢球的环境，目前会加载机器人以及足球
            > 变量含义：
            > - `qpos`和`qvel`：机器人关节位置和速度

    - `logs/` \
        tensorboard logs
    - `models/`\
        存训练好的RL模型
    - `test_urdf.py`\
        可用来测试URDF以及仿真环境数据读取
    - `train.py`\
        使用stable-baselines3来进行RL训练
    - `config.py`\
        存放机器人和仿真环境的参数

# TODO
1. friction, max_torque, max_velocity
2. p.setJointMotorControl2中的positionGain
