import torch
import torch.nn.functional as F
import numpy as np

class DQNAgent:
    def __init__(self, model, target_model, optimizer, gamma):
        self.model = model
        self.target_model = target_model
        self.optimizer = optimizer
        self.gamma = gamma

    def act(self, state, epsilon, action_space):
        if np.random.rand() < epsilon:
            return action_space.sample()
        with torch.no_grad():
            return self.model(state).argmax().item()

    def train_step(self, batch):
        states, actions, rewards, next_states, dones = batch
        q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze()
        with torch.no_grad():
            q_next = self.target_model(next_states).max(1)[0]
            target = rewards + self.gamma * q_next * (1 - dones)

        loss = F.mse_loss(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
