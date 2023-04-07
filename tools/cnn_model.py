import torch
import torch.nn as nn
import torch.nn.functional as F
from config import config

class DenseBlock(nn.Module):
    def __init__(self, num=4):
        super(DenseBlock, self).__init__()
        self.block = nn.Sequential()
        self.make_block(num)

    def DenseLayers(self, i):
        return nn.Conv2d(in_channels=i * 32, out_channels=32, kernel_size=3, stride=1, padding=1).cuda()

    def make_block(self, num):
        for i in range(1, num):
            layer = self.DenseLayers(i)
            self.block.add_module('c_'+str(i),layer)

    def forward(self, x):
        feature = [x]
        for name, layer in self.block.named_children():
            x = F.tanh(layer(torch.cat(feature, 1)))
            feature.append(x)
        return x


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.block = DenseBlock(num=7)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=2, padding=0)
        self.fc1 = nn.Linear(1600, 128)
        self.fc_pi = nn.Linear(128, config['act_dim'])

    def pi(self, x, softmax_dim=-1):
        x = F.tanh(self.conv1(x))
        x = self.block(x)
        x = F.tanh(self.conv2(x))
        x = F.tanh(self.conv3(x))
        x = x.view(-1, 1600)
        x = F.tanh(self.fc1(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob