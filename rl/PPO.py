import gymnasium
import quadcopter
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import torch


ENV_NAME = "Quadcopter-v0"
SEED = 42

vec_env = make_vec_env(ENV_NAME, n_envs=3, seed=SEED)  # Parallel environments
model = PPO(
    "MlpPolicy",
    vec_env,
    verbose=1,
    tensorboard_log=f"logs",
    policy_kwargs=dict(
        activation_fn=torch.nn.ReLU, net_arch=dict(pi=[256, 256], vf=[256, 256])
    ),
    learning_rate=0.00005,
    clip_range=0.05,
    seed=SEED,
    batch_size=256,
    max_grad_norm=0.2,
)
model.learn(total_timesteps=20_000_000)
model.save(f"./policy/PPO_{ENV_NAME}")
del model
vec_env.close()
