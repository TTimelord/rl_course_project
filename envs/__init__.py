from gym.envs.registration import register
from envs.kick_ball_env import KickBall

register(
     id="KickBall-v0",
     entry_point="envs:KickBall",
)