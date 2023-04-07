import torch.nn as nn
import torch.nn.functional as F
from config import config
from torchvision.models import vgg16


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.bone = vgg16().features
        self.bone[0] = nn.Conv2d(4, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.avg = nn.AdaptiveAvgPool2d((2, 2))
        self.fc_1 = nn.Linear(2048, 512)
        self.fc_pi = nn.Linear(512, config['act_dim'])
        self.fc_v = nn.Linear(512, 1)

    def pi(self, x):
        x = self.bone(x)
        x = self.avg(x)
        x = F.relu(self.fc_1(x.view(-1, 2048)))
        x = F.softmax(self.fc_pi(x), dim=-1)
        return x

    def v(self, x):
        x = self.bone(x)
        x = self.avg(x)
        x = F.relu(self.fc_1(x.view(-1, 2048)))
        v = self.fc_v(x)
        return v
