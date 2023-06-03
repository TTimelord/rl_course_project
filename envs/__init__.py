from gym.envs.registration import register
from envs.kick_ball_env import KickBall
from envs.kick_ball_env_wq import KickBall57
from envs.run_env import Run

register(
     id="KickBall-v0",
     entry_point="envs:KickBall",
)
register(
     id="Run-v0",
     entry_point="envs:Run",
)
register(
     id="KickBall57-v0",
     entry_point="envs:KickBall57",
)