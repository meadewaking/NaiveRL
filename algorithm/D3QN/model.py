import torch.nn as nn
import torch.nn.functional as F
from config import config


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(config['env_dim'], 128)
        self.fc2_Q = nn.Linear(128, config['act_dim'])
        self.fc2_val = nn.Linear(128, 1)

    def forward(self, x):
        As = self.fc2_Q(F.relu(self.fc1(x)))
        V = self.fc2_val(F.relu(self.fc1(x)))
        Q = As + (V - As.mean(dim=-1, keepdim=True))

        return Q
