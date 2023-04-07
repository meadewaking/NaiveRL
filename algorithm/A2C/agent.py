import torch
import numpy as np
from config import config

torch.set_num_threads(1)


class Agent():
    def __init__(self, algorithm):
        self.alg = algorithm
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def sample(self, state):
        state = torch.tensor(state, device=self.device, dtype=torch.float)
        action = self.alg.sample(state)
        return action

    def predict(self, state):
        state = torch.tensor(state, device=self.device, dtype=torch.float)
        action = self.alg.predict(state)
        return action

    def learn(self, states, actions, rewards, mask_lst):
        # states = torch.tensor(states, dtype=torch.float)
        # actions = np.expand_dims(actions, 1)
        # actions = torch.tensor(actions, dtype=torch.long)
        states = np.asarray(states)
        actions = np.expand_dims(actions, 1)
        rewards = np.clip(rewards, -1, 1) / config['reward_scale']
        mask_lst = np.asarray(mask_lst)
        td_target = self.compute_target(rewards, mask_lst)
        loss = self.alg.learn(states, actions, td_target)

        return loss

    def compute_target(self, rewards, mask_lst):
        G = 0
        td_target = []
        for r, mask in zip(rewards[::-1], mask_lst[::-1]):
            G = r + self.alg.gamma * G * mask
            td_target.append(G)
        td_target = np.asarray(td_target[::-1])
        return td_target
