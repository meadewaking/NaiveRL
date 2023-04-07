import torch
import numpy as np

torch.set_num_threads(1)


class Agent():
    def __init__(self, algorithm):
        self.alg = algorithm

    def sample(self, state):
        state = torch.tensor(state, dtype=torch.float)
        action = self.alg.sample(state)
        return action

    def predict(self, state):
        state = torch.tensor(state, dtype=torch.float)
        action = self.alg.predict(state)
        return action

    def learn(self, states, actions, rewards, local_model, s_, done):
        states = torch.tensor(states, dtype=torch.float)
        actions = np.expand_dims(actions, 1)
        actions = torch.tensor(actions, dtype=torch.long)
        s_ = torch.tensor(s_, dtype=torch.float)
        loss = self.alg.learn(states, actions, rewards, local_model, s_, done)

        return loss
