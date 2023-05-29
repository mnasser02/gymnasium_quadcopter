import gymnasium as gym
import quadcopter
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
import datetime

ENV_NAME = "Quadcopter-v0"
SEED = 42

model = PPO.load(
    f"./policy/PPO_{ENV_NAME}",
    tensorboard_log="logs",
)
vec_env = make_vec_env(ENV_NAME, n_envs=3, seed=SEED)
model.set_env(vec_env)

model.learn(total_timesteps=10_000_000, reset_num_timesteps=False)

now = datetime.now().strftime("%Y-%m-%d_%H:%M")
model.save(f"./policy/PPO_{ENV_NAME}_{now}")
del model
vec_env.close()
