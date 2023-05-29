import gymnasium as gym
from stable_baselines3 import PPO
import quadcopter

ENVIRONMENT = "Quadcopter-v0"

env = gym.make(ENVIRONMENT, render_mode="human")

obs, info = env.reset()
while True:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    # print(obs[:3])
    env.render()
    if terminated or truncated:
        obs = env.reset()

env.close()
