import torch.nn as nn
import torch.nn.functional as F
from config import config


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(config['env_dim'], 128)
        self.fc_pi = nn.Linear(128, config['act_dim'])
        self.fc_v = nn.Linear(128, 1)

    def forward(self, x):
        return F.relu(self.fc1(x))

    def pi_logits(self, x):
        x = self.forward(x)
        return self.fc_pi(x)

    def pi(self, x):
        return F.softmax(self.pi_logits(x), dim=-1)

    def v(self, x):
        x = self.forward(x)
        return self.fc_v(x)

    def pi_v(self, x):
        x = self.forward(x)
        return self.fc_pi(x), self.fc_v(x)
