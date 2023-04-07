import numpy as np
import torch
from config import config


class Agent():
    def __init__(self, algorithm):
        self.act_dim = config['act_dim']
        self.exploration = config['exploration_rate']
        self.global_step = 0
        self.update_target_steps = config['update_interval']
        self.alg = algorithm
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def sample(self, state):
        sample = np.random.random()
        if sample < self.exploration:
            action = np.random.randint(self.act_dim)
        else:
            action = self.predict(state)
        self.exploration = max(config['last_expr'], self.exploration - config['expr_step'])
        return action

    def predict(self, state):
        state = np.expand_dims(state, 0)
        state = torch.tensor(state, dtype=torch.float, device=self.device)
        action = self.alg.predict(state)
        return action

    def learn(self, states, actions, rewards, next_states, dones):
        if self.global_step % self.update_target_steps == 0:
            self.alg.update()
        self.global_step += 1

        actions = np.expand_dims(actions, -1)
        dones = np.expand_dims(dones, -1)
        rewards = np.expand_dims(rewards, -1)
        rewards = np.clip(rewards, -1, 1)

        states = torch.tensor(states, dtype=torch.float, device=self.device)
        next_states = torch.tensor(next_states, dtype=torch.float, device=self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float, device=self.device)

        loss = self.alg.learn(states, actions, rewards, next_states, dones)
        return loss
