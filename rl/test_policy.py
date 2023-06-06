import gymnasium as gym
from stable_baselines3 import PPO
import quadcopter

ENVIRONMENT = "Quadcopter-v0"

env = gym.make(ENVIRONMENT, render_mode="human", max_time_steps=400)
model = PPO.load("./policy/best_model")

obs, info = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    # print(obs[:3])
    env.render()
    if terminated or truncated:
      obs, info = env.reset()

env.close()
