import gymnasium as gym
import quadcopter
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
import datetime

ENV_NAME = "Quadcopter-v0"
SEED = 123

model = PPO.load(
    f"./policy2/best_model",
    tensorboard_log="logs",
)
vec_env = make_vec_env(ENV_NAME, n_envs=8, seed=SEED)
model.set_env(vec_env)

# Separate evaluation env
eval_env = gym.make(ENV_NAME)
# Use deterministic actions for evaluation
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./policy4",
    log_path="logs",
    eval_freq=500,
    deterministic=True,
    render=False,
)

model.learn(total_timesteps=20_000_000, reset_num_timesteps=False, callback=eval_callback)

del model
vec_env.close()
