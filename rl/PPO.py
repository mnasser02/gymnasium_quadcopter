import gymnasium as gym
import quadcopter
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
import torch
import datetime

ENV_NAME = "Quadcopter-v0"
SEED = 200

vec_env = make_vec_env(ENV_NAME, n_envs=6, seed=SEED)  # Parallel environments
model = PPO(
    "MlpPolicy",
    vec_env,
    verbose=1,
    tensorboard_log="logs",
    policy_kwargs=dict(
        activation_fn=torch.nn.ReLU, net_arch=dict(pi=[256, 256], vf=[256, 256])
    ),
    learning_rate=0.00005,
    gamma=0.99,
    clip_range=0.05,
    seed=SEED,
    batch_size=256,
)

# Separate evaluation env
eval_env = gym.make(ENV_NAME)
# Use deterministic actions for evaluation
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./policy2",
    log_path="logs",
    eval_freq=500,
    deterministic=True,
    render=False,
)

model.learn(total_timesteps=20_000_000, callback=eval_callback)

now = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")
model.save(f"./policy/PPO_{ENV_NAME}_{now}")
del model
vec_env.close()
