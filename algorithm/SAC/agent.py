import numpy as np
import torch
from config import config


class Agent():
    def __init__(self, algorithm):
        self.act_dim = config['act_dim']
        self.alg = algorithm
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def predict(self, state):
        state = np.expand_dims(state, 0)
        state = torch.tensor(state, dtype=torch.float, device=self.device)
        action = self.alg.predict(state)
        return [action.cpu().item() * config['action_scale']]

    def sample(self, state):
        state = np.expand_dims(state, 0)
        state = torch.tensor(state, dtype=torch.float, device=self.device)
        action, _ = self.alg.sample(state)
        return [action.cpu().item() * config['action_scale']]

    def learn(self, states, actions, rewards, next_states, dones):
        dones = np.expand_dims(dones, -1)
        rewards = np.expand_dims(rewards, -1) / config['reward_scale']
        # rewards = np.clip(rewards, -1, 1)

        states = torch.tensor(states, dtype=torch.float, device=self.device)
        next_states = torch.tensor(next_states, dtype=torch.float, device=self.device)
        actions = torch.tensor(actions, dtype=torch.float, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float, device=self.device)

        loss = self.alg.learn(states, actions, rewards, next_states, dones)
        self.alg.soft_update()
        return loss
