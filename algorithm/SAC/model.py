import torch
import torch.nn as nn
import torch.nn.functional as F
from config import config


class Pi(nn.Module):
    def __init__(self):
        super(Pi, self).__init__()
        self.fc1 = nn.Linear(config['state_dim'], 128)
        self.fc_mu = nn.Linear(128, config['act_dim'])
        self.fc_std = nn.Linear(128, config['act_dim'])

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = self.fc_mu(x)
        std = self.fc_std(x)
        return mu, std


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
