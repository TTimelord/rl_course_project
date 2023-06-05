from gym.envs.registration import register
from envs.kick_ball_env import KickBall
from envs.kick_ball_stand_env import KickBallStand
from envs.run_env import Run

register(
     id="KickBall-v0",
     entry_point="envs:KickBall",
)
register(
     id="KickBallStand-v0",
     entry_point="envs:KickBallStand",
)
register(
     id="Run-v0",
     entry_point="envs:Run",
)