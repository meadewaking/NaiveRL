import torch
import numpy as np

torch.set_num_threads(1)


class Agent():
    def __init__(self, algorithm):
        self.alg = algorithm
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def sample(self, state):
        state = torch.as_tensor(np.asarray(state), device=self.device, dtype=torch.float)
        action, log_prob = self.alg.sample(state)
        return action, log_prob

    def predict(self, state):
        state = torch.as_tensor(np.asarray(state), device=self.device, dtype=torch.float)
        action = self.alg.predict(state)
        return action

    def learn(self, trajectories):
        loss = self.alg.learn(trajectories)
        return loss.item()
