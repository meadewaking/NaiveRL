import torch
import numpy as np


class Agent():
    def __init__(self, algorithm):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.alg = algorithm

    def sample(self, state):
        state = torch.as_tensor(np.asarray(state), device=self.device, dtype=torch.float)
        action = self.alg.sample(state)
        return action

    def predict(self, state):
        state = torch.as_tensor(np.asarray(state), device=self.device, dtype=torch.float)
        action = self.alg.predict(state)
        return action

    def learn(self, states, actions, rewards, s_, done):
        states = torch.as_tensor(np.asarray(states), device=self.device, dtype=torch.float)
        actions = np.expand_dims(actions, 1)
        actions = torch.as_tensor(actions, device=self.device, dtype=torch.long)
        rewards = torch.as_tensor(np.asarray(rewards), device=self.device, dtype=torch.float).view(-1)
        s_ = torch.as_tensor(np.asarray(s_), device=self.device, dtype=torch.float)
        
        loss = self.alg.learn(states, actions, rewards, s_, done)
        return loss.item()
