import torch
import torch.optim as optim
from bdqn.models import QNetwork, BetaNetwork
from replay_buffer import ReplayBuffer

class BetaDQNAgent:
    def __init__(self, obs_space, action_space, config):
        self.device = config.device

        self.q_net = QNetwork(obs_space, action_space).to(self.device)
        self.target_q_net = QNetwork(obs_space, action_space).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.beta_net = BetaNetwork(obs_space, action_space).to(self.device)

        self.optimizer_q = optim.Adam(self.q_net.parameters(), lr=config.lr_q)
        self.optimizer_beta = optim.Adam(self.beta_net.parameters(), lr=config.lr_beta)

        self.replay = ReplayBuffer(config.buffer_size, obs_space, action_space)

        self.gamma = config.gamma
        self.batch_size = config.batch_size
        self.update_target_every = config.update_target_every
        self.step_count = 0

    def act(self, state, policy):
        with torch.no_grad():
            qvals = self.q_net(torch.tensor(state).to(self.device))
            beta = self.beta_net(torch.tensor(state).to(self.device))

        return policy(qvals, beta)

    def store_transition(self, s, a, r, s2, done):
        self.replay.push(s, a, r, s2, done)

    def train_step(self):
        if len(self.replay) < self.batch_size:
            return

        batch = self.replay.sample(self.batch_size)
        
        q_loss = compute_td_loss(self.q_net, self.target_q_net, batch, self.gamma)

        beta_loss = compute_beta_loss(self.beta_net, batch)

        self.optimizer_q.zero_grad()
        q_loss.backward()
        self.optimizer_q.step()

        self.optimizer_beta.zero_grad()
        beta_loss.backward()
        self.optimizer_beta.step()

        self.step_count += 1
        if self.step_count % self.update_target_every == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
