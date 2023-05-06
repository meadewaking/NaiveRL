import torch
import numpy as np


class Agent():
    def __init__(self, algorithm):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.alg = algorithm

    def sample(self, state):
        state = torch.tensor(state, device=self.device, dtype=torch.float)
        action = self.alg.sample(state)
        return action

    def predict(self, state):
        state = torch.tensor(state, device=self.device, dtype=torch.float)
        action = self.alg.predict(state)
        return action

    def learn(self, states, actions, rewards, s_, done):
        states = np.asarray(states)
        states = torch.tensor(states, device=self.device, dtype=torch.float)
        actions = np.expand_dims(actions, 1)
        actions = torch.tensor(actions, device=self.device, dtype=torch.long)
        rewards = np.expand_dims(rewards, 1)
        s_ = torch.tensor(s_, device=self.device, dtype=torch.float)
        # loss = self.alg.learn(states, actions, rewards, s_, done)
        loss = self.alg.learn_v3(states, actions, rewards, s_, done)
        return loss.item()
