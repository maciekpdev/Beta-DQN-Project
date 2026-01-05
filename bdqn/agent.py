import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from bdqn.models import QNetwork, BetaNetwork

class BetaDQNAgent:
    def __init__(self, obs_space, action_space, config):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.q_net = QNetwork(obs_space, action_space).to(self.device)
        self.target_q_net = QNetwork(obs_space, action_space).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.beta_net = BetaNetwork(obs_space, action_space).to(self.device)

        self.optimizer_q = optim.Adam(self.q_net.parameters(), lr=config.get("lr_q", 1e-3))
        self.optimizer_beta = optim.Adam(self.beta_net.parameters(), lr=config.get("lr_beta", 1e-3))

        self.gamma = config.get("gamma", 0.99)
        self.update_target_every = config.get("update_target_every", 64)
        self.step_count = 0

        self.criterion_beta = nn.CrossEntropyLoss()
        self.criterion_q = nn.SmoothL1Loss()

    def act(self, state, epsilon, policy):
        with torch.no_grad():
            qvals = self.q_net(torch.tensor(state).to(self.device))
            beta = self.beta_net(torch.tensor(state).to(self.device))
            
        return policy(qvals, beta, epsilon)

    def train_step(self, batch):
        states, actions, rewards, next_states, dones = batch

        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(np.array(actions)).long().to(self.device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        dones = torch.tensor(np.array(dones), dtype=torch.float32).to(self.device)

        q_loss = self.train_td(states, actions, next_states, rewards, dones)
        beta_loss = self.train_beta_net(states, actions)

        self.step_count += 1
        if self.step_count % self.update_target_every == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())

        return q_loss, beta_loss

    def train_td(self, states, actions, next_states, rewards, dones):
        q = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze()
        with torch.no_grad():
            q_next = self.target_q_net(next_states).max(1)[0]
            target = rewards + self.gamma * q_next * (1 - dones)
        loss = self.criterion_q(q, target)
        self.optimizer_q.zero_grad()
        loss.backward()
        self.optimizer_q.step()

        return loss.item()

    def train_beta_net(self, states, actions):
        logits = self.beta_net(states)
        loss = self.criterion_beta(logits, actions)
        self.optimizer_beta.zero_grad()
        loss.backward()
        self.optimizer_beta.step()

        return loss.item()
    
    def save_checkpoint(self):
        checkpoint_path = f"checkpoints/bqn.pt"

        torch.save(self.q_net.state_dict(), checkpoint_path)
        print(f"Training finished. Model saved to {checkpoint_path}")