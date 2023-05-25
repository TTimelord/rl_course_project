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
        存放强化学习 gym 环境(目前使用的stable-baselines3是1.8.0版本，因此还使用的是gym而不是gymnasium(https://gymnasium.farama.org/))。
        - `kick_ball_env.py`
        踢球的环境，目前会加载机器人以及足球。
            > 变量含义：
            > - `qpos`和`qvel`：机器人关节位置和速度
            > - `goal_pos`: 球门的位置
            > - `pos`: 机器人身体位置
        - `run_env.py`
        一个继承自KickBall的环境，用来学习跑步，可以使用类似的方法写新环境

    - `logs/` \
        tensorboard logs
    - `models/`\
        存训练好的RL模型
    - `test_urdf.py`\
        可用来测试URDF以及仿真环境数据读取
    - `train.py`\
        使用stable-baselines3来进行RL训练
    - `test.py`\
        测试训练出来的RL模型
    - `config.py`\
        存放机器人和仿真环境的参数

# TODO
## 确定参数
1. friction, max_torque, max_velocity
2. p.setJointMotorControl2中的positionGain
## 奖励函数
1. contact reward 需要重新检查link序号
## Randomization
球要随机放置吗
