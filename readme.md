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
        gym environments
        - `kick_ball_env.py` only task reward
        - `kick_ball_stand_env.py` task reward with fall termination
        - `kick_ball_imitation_env` task reward, termination and imitation reward
    - `logs/` \
        tensorboard logs
    - `models/`\
        saved models
    - `test_urdf.py`\
        test simulation and urdf
    - `train.py`\
        use stable-baselines3 to train RL agnet
    - `test.py`\
        Test trained model
    - `config.py`\
        parameters of the robot and simulation
