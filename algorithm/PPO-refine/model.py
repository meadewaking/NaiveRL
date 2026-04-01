import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, obs_dim=4, act_dim=2, hidden=128):
        """
        PPO Model (Optimized Version)
        
        Differences from original:
        - Added pi_v() method for shared forward computation
        - Extensible for visual inputs (see PPO_human_pre for VGG16 backbone)
        """
        super(Model, self).__init__()
        self.fc1 = nn.Linear(obs_dim, hidden)
        self.fc_pi = nn.Linear(hidden, act_dim)
        self.fc_v = nn.Linear(hidden, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return x

    def pi(self, x):
        x = self.forward(x)
        return F.softmax(self.fc_pi(x), dim=-1)

    def v(self, x):
        x = self.forward(x)
        return self.fc_v(x)

    # ============ Optimization 1: Shared Forward Pass ============
    # Avoid redundant fc1 computation in pi() and v()
    def pi_v(self, x):
        x = self.forward(x)
        return F.softmax(self.fc_pi(x), dim=-1), self.fc_v(x)
    
    # ============ Extensible: Visual Input Version ===============
    # For image-based inputs, use architecture similar to PPO_human_pre:
    # def __init__(self):
    #     super().__init__()
    #     self.bone = vgg16().features  # Shared visual backbone
    #     self.avg = nn.AdaptiveAvgPool2d((2, 2))
    #     self.fc_shared = nn.Linear(2048, 512)
    #     self.fc_pi = nn.Linear(512, act_dim)
    #     self.fc_v = nn.Linear(512, 1)
    #
    # def pi_v(self, x):
    #     x = self.bone(x)
    #     x = self.avg(x)
    #     x = F.relu(self.fc_shared(x.view(-1, 2048)))
    #     return F.softmax(self.fc_pi(x), dim=-1), self.fc_v(x)