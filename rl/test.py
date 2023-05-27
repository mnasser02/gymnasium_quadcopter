import gymnasium as gym
from stable_baselines3 import PPO
import quadcopter

ENVIRONMENT = "Quadcopter-v0"

env = gym.make(ENVIRONMENT, render_mode="human", reset_info=True, random_target=False)
model = PPO.load(f"PPO_{ENVIRONMENT}")

obs, info = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    # print(obs[:3])
    env.render()
    # VecEnv resets automatically
    # if done:
    #   obs = env.reset()

env.close()
