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
        rewards = np.transpose(rewards, (1, 0))
        s_ = torch.tensor(s_, device=self.device, dtype=torch.float)
        states = states.permute(1, 0, 2, 3, 4)
        actions = actions.permute(2, 0, 1)
        loss = self.alg.learn(states, actions, rewards, s_, done)
        return loss.item()
