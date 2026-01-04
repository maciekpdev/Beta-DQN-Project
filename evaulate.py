import argparse
import gymnasium as gym
import torch
from dqn.model import DQN
import yaml
import numpy as np

import argparse

parser = argparse.ArgumentParser(description="Evaluate a trained DQN agent")
parser.add_argument(
    "--checkpoint",
    type=str,
    default="checkpoints/dqn_MountainCar-v0.pt",
    help="Path to the saved model checkpoint (.pt)"
)
parser.add_argument(
    "--env",
    type=str,
    default="MountainCar-v0",
    help="Gym environment name"
)
args = parser.parse_args()

checkpoint_path = args.checkpoint
state_dict = torch.load(checkpoint_path)

env_name = args.env

env = gym.make(env_name, render_mode="human")
obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

model = DQN(obs_dim, action_dim)
model.load_state_dict(state_dict)
model.eval()

obs, _ = env.reset()
done = False

while not done:
    with torch.no_grad():
        obs_tensor = torch.tensor(np.array(obs), dtype=torch.float32).unsqueeze(0)
        action = model(obs_tensor).argmax().item()
    obs, _, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
