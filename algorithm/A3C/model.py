import torch.nn as nn
import torch.nn.functional as F
from config import config


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(config['state_dim'], 100)
        self.fc_pi = nn.Linear(100, config['act_dim'])
        self.fc_v = nn.Linear(100, 1)

    def pi(self, x, softmax_dim=-1):
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob

    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v
