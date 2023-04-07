import torch.nn as nn
import torch.nn.functional as F
from config import config


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(3872, 465)
        self.fc2_Q = nn.Linear(465, config['act_dim'])
        self.fc2_val = nn.Linear(465, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc1(x.view(-1, 3872)))
        As = self.fc2_Q(x)
        V = self.fc2_val(x)
        Q = As + (V - As.mean(dim=-1, keepdim=True))

        return Q
