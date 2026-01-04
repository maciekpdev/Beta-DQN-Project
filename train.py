import argparse
import gymnasium as gym
import torch
import torch.optim as optim
import numpy as np
import yaml
from dqn.model import DQN
from dqn.replay_buffer import ReplayBuffer
from dqn.agent import DQNAgent
from dqn.utils import set_seed

parser = argparse.ArgumentParser(description="Train a DQN agent with a YAML config")
parser.add_argument("config", type=str, help="Path to YAML config file")
args = parser.parse_args()

with open("configs/" + args.config, "r") as f:
    config = yaml.safe_load(f)

env_name = config.get("env", "")

if not env_name:
    raise SystemExit("No environment name specified, program stopped")

gamma = config.get("gamma", 0.99)
lr = config.get("lr", 1e-3)
batch_size = config.get("batch_size", 64)
episodes = config.get("episodes", 500)
epsilon_start = config.get("epsilon_start", 1.0)
epsilon_end = config.get("epsilon_end", 0.05)
epsilon_decay = config.get("epsilon_decay", 0.995)
seed = config.get("seed", 42)

env = gym.make(env_name)
print("Learning on " + env_name)
obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

print(obs_dim, " <-- obs dim", action_dim, " <-- action dim")

device = "cuda" if torch.cuda.is_available() else "cpu"

set_seed(seed)
env.reset(seed=seed)

policy_net = DQN(obs_dim, action_dim).to(device)
target_net = DQN(obs_dim, action_dim).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.Adam(policy_net.parameters(), lr=lr)
agent = DQNAgent(policy_net, target_net, optimizer, gamma)

buffer = ReplayBuffer()

epsilon = epsilon_start
target_update = 500
step = 0

for episode in range(episodes):
    obs, _ = env.reset()
    total_reward = 0
    episode_loss = 0.0
    loss_count = 0

    while True:
        step += 1
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)

        action = agent.act(obs_tensor, epsilon, env.action_space)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        buffer.push(obs, action, reward, next_obs, done)
        obs = next_obs
        total_reward += reward

        if len(buffer) > batch_size:
            batch_data = buffer.sample(batch_size)
            states, actions, rewards, next_states, dones = batch_data

            states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
            actions = torch.tensor(np.array(actions)).long().to(device)
            rewards = torch.tensor(np.array(rewards), dtype=torch.float32).to(device)
            next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(device)
            dones = torch.tensor(np.array(dones), dtype=torch.float32).to(device)

            loss_value = agent.train_step((states, actions, rewards, next_states, dones))
            episode_loss += loss_value
            loss_count += 1

        if step % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if done:
            break

    epsilon = max(epsilon * epsilon_decay, epsilon_end)
    avg_loss = episode_loss / max(loss_count, 1)

    print(
        f"Episode {episode}, "
        f"Reward: {total_reward:.1f}, "
        f"Avg Loss: {avg_loss:.4f}, "
        f"Epsilon: {epsilon:.3f}"
    )


checkpoint_path = f"checkpoints/dqn_{env_name}.pt"

torch.save(policy_net.state_dict(), checkpoint_path)
print(f"Training finished. Model saved to {checkpoint_path}")
