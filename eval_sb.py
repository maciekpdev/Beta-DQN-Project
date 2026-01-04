import gymnasium as gym
from stable_baselines3 import DQN

env = gym.make("MountainCar-v0", render_mode="human")
model = DQN.load("sb3_dqn_mountaincar")

obs, _ = env.reset()
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, _, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
