import copy
import torch
import torch.nn as nn
import torch.optim as optim
from config import config


class Alg():
    def __init__(self, model):
        self.model = model
        self.target_model = copy.deepcopy(self.model)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.target_model.to(self.device)
        self.gamma = config['gamma']
        self.lr = config['learning_rate']

        self.mse_loss = torch.nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def predict(self, state):
        with torch.no_grad():
            pred_q = self.model(state).cpu()
        action = pred_q.argmax(dim=-1).numpy()
        return action

    def learn(self, states, actions, rewards, next_states, done):
        pred_value = self.model(states).gather(1, actions)
        greedy_action = self.model(next_states).max(dim=-1, keepdim=True)[1]
        with torch.no_grad():
            max_v = self.target_model(next_states).gather(1, greedy_action)
            target = rewards + (1 - done) * self.gamma * max_v
        self.optimizer.zero_grad()
        loss = self.mse_loss(pred_value, target)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def update(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def drop_reset(self, p=0.01):  # 将p部分的网络重置
        for name, h in self.model.named_children():
            if type(h) == nn.Linear:
                m = copy.deepcopy(h)
                m.requires_grad_(False)
                reset_matrix = copy.deepcopy(m)
                mask = (torch.rand(h.weight.shape) > p).float()
                anti_mask = 1 - mask
                mask = mask.clone().detach().to(self.device)
                anti_mask = anti_mask.clone().detach().to(self.device)
                torch.nn.init.normal_(reset_matrix.weight, std=0.01)
                m.weight = torch.nn.Parameter(m.weight * mask, requires_grad=False)
                reset_matrix.weight = torch.nn.Parameter(reset_matrix.weight * anti_mask, requires_grad=False)
                m.weight += reset_matrix.weight
                eval('self.model.' + name + '.load_state_dict(m.state_dict())')
