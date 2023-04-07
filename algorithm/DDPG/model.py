import torch
import torch.nn as nn
import torch.nn.functional as F
from config import config


class Pi(nn.Module):
    def __init__(self):
        super(Pi, self).__init__()
        self.fc1 = nn.Linear(config['state_dim'], 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc_pi = nn.Linear(64, config['act_dim'])

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        pi = torch.tanh(self.fc_pi(x)) * config['action_scale']
        return pi


class Q(nn.Module):
    def __init__(self):
        super(Q, self).__init__()
        self.fc_s = nn.Linear(config['state_dim'], 64)
        self.fc_a = nn.Linear(config['act_dim'], 64)
        self.fc_1 = nn.Linear(128, 32)
        self.fc_q = nn.Linear(32, 1)

    def forward(self, x, a):
        h1 = F.relu(self.fc_s(x))
        h2 = F.relu(self.fc_a(a))
        cat = torch.cat([h1, h2], dim=-1)
        q = F.relu(self.fc_1(cat))
        q = self.fc_q(q)
        return q
