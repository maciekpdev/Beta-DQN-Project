import argparse
import gymnasium as gym
import torch
import torch.optim as optim
import numpy as np
import yaml
from dqn.model import DQN
from dqn.replay_buffer import ReplayBuffer
from bdqn.agent import BetaDQNAgent
from bdqn.meta_controller import MetaController
from utils import set_seed
from bdqn.policies import CorPolicy, CovPolicy
from bdqn.replay_buffer import ReplayBuffer
import bdqn.policies

#parser = argparse.ArgumentParser(description="Train a DQN agent with a YAML config")
#parser.add_argument("config", type=str, help="Path to YAML config file")
#args = parser.parse_args()

with open("configs/betadqn.yaml", "r") as f:
    config = yaml.safe_load(f)

env_name = config.get("env", "")

if not env_name:
    raise SystemExit("No environment name specified, program stopped")

env = gym.make(env_name)
print("Learning on " + env_name)
obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

print(obs_dim, " <-- obs dim", action_dim, " <-- action dim")
seed = config.get("seed", 42)
set_seed(seed)
env.reset(seed=seed)

agent = BetaDQNAgent(obs_dim, action_dim, config)
1
cor_policies = [CorPolicy(1/10) for i in range(1, 10)]
cov_policies = [CovPolicy(0.05), CovPolicy(0.1)]

policies = cor_policies + cov_policies

controller = MetaController(len(policies))

buffer = ReplayBuffer()
episodes = config.get("episodes", 500)
batch_size = config.get("batch_size", 64)

epsilon_start = config.get("epsilon_start", 1.0)
epsilon_end = config.get("epsilon_end", 0.05)
epsilon_decay = config.get("epsilon_decay", 0.995)
epsilon = epsilon_start

target_update = 500
step = 0

replay_buffer = ReplayBuffer()

for episode in range(episodes):
    obs, _ = env.reset()
    total_reward = 0
    total_exploration_steps = 0
    q_loss_count = 0
    beta_loss_count = 0
    episode_beta_loss = 0.0
    episode_q_loss = 0.0

    policy = controller.select_policy()

    steps_per_episode = 0

    while True:
        steps_per_episode += 1
        step += 1

        action, is_exploration_move = agent.act(obs, epsilon, policies[policy])

        if is_exploration_move:
            total_exploration_steps += 1

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        replay_buffer.push(obs, action, reward, next_obs, done)
        obs = next_obs
        total_reward += reward

        if len(replay_buffer) > batch_size:
            batch = replay_buffer.sample(batch_size)
            q_loss, beta_loss = agent.train_step(batch)
            episode_beta_loss += beta_loss
            episode_q_loss += q_loss_count
            q_loss_count += 1
            beta_loss_count += 1

        if done:
            break

    controller.update(policy, total_reward, total_exploration_steps / steps_per_episode)

    epsilon = max(epsilon * epsilon_decay, epsilon_end)
    avg_q_loss = episode_q_loss / max(q_loss_count, 1)
    avg_beta_loss = episode_beta_loss / max(beta_loss_count, 1)

    print(
        f"Episode {episode}, "
        f"Reward: {total_reward:.1f}, "
        f"Avg Loss: {avg_q_loss:.4f}, "
        f"Beta Loss: {avg_beta_loss:.4f}, "
        f"Epsilon: {epsilon:.3f}"
    )

agent.save_checkpoint()
