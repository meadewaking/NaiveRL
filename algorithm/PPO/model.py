import torch.nn as nn
import torch.nn.functional as F
from config import config


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(config['env_dim'], 128)
        self.fc_pi = nn.Linear(128, config['act_dim'])
        self.fc_v = nn.Linear(128, 1)

    def pi(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc_pi(x), dim=-1)
        return x

    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v

    def pi_v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        x = F.softmax(self.fc_pi(x), dim=-1)
        return x, v
