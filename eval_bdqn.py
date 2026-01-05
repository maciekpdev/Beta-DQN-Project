import torch
import gymnasium as gym
import numpy as np

from bdqn.agent import BetaDQNAgent
from config import config

env = gym.make("MountainCar-v0")

agent = BetaDQNAgent(
    obs_space=env.observation_space.shape[0],
    action_space=env.action_space.n,
    config=config
)

checkpoint = torch.load("checkpoints/bqn.pt", map_location=agent.device)
agent.q_net.load_state_dict(checkpoint)

agent.q_net.eval()
agent.beta_net.eval()

obs, _ = env.reset()
done = False
total_reward = 0

while not done:
    obs_tensor = torch.tensor(obs, dtype=torch.float32).to(agent.device)

    with torch.no_grad():
        q_vals = agent.q_net(obs_tensor)
        action = q_vals.argmax().item()

    obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    total_reward += reward

env.close()
print("Episode reward:", total_reward)
