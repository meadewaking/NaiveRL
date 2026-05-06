import numpy as np
import torch


class Agent():
    def __init__(self, algorithm):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.alg = algorithm

    def sample(self, state):
        state = torch.as_tensor(np.asarray(state), device=self.device, dtype=torch.float)
        return self.alg.sample(state)

    def predict(self, state):
        state = torch.as_tensor(np.asarray(state), device=self.device, dtype=torch.float)
        return self.alg.predict(state)

    def learn(self, states, actions, rewards, s_, terminal):
        states = torch.as_tensor(np.asarray(states), device=self.device, dtype=torch.float)
        actions = torch.as_tensor(np.asarray(actions), device=self.device, dtype=torch.long)
        rewards = torch.as_tensor(np.asarray(rewards), device=self.device, dtype=torch.float)
        s_ = torch.as_tensor(np.asarray(s_), device=self.device, dtype=torch.float)
        loss = self.alg.learn(states, actions, rewards, s_, terminal)
        return loss.item()
