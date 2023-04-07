import torch
from torch.distributions import Categorical
from config import config
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np


class Alg():
    def __init__(self, model):
        self.model = model
        self.learning_rate = config['learning_rate']
        self.gamma = config['gamma']
        self.entropy_coeff = config['entropy_coeff']
        self.vf_loss_coeff = config['vf_loss_coeff']
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.lr_scheduler = lr_scheduler.StepLR(self.optimizer, config['lr_step'], config['lr_gamma'])

    def learn(self, states, actions, rewards, local_model, s_, done):
        s_final = torch.unsqueeze(s_, 0)
        R = 0.0 if done else local_model.v(s_final).item()
        td_target_lst = []
        r_lst = np.clip(rewards, -1, 1)/config['reward_scale']
        for reward in r_lst[::-1]:
            R = self.gamma * R + reward
            td_target_lst.append([R])
        td_target_lst.reverse()
        td_target = torch.tensor(td_target_lst.copy(), dtype=torch.float)
        advantage = td_target - local_model.v(states)

        pi = local_model.pi(states)
        actions_log_probs = torch.log(pi.gather(1, actions) + 1e-8)
        policy_loss = -((actions_log_probs * advantage.detach()).sum())
        value_delta = local_model.v(states) - td_target.detach()
        value_loss = torch.mul(value_delta, value_delta).sum()
        entropy_loss = (-torch.log(pi + 1e-8) * torch.exp(torch.log(pi + 1e-8))).sum()
        loss = policy_loss + self.vf_loss_coeff * value_loss + self.entropy_coeff * entropy_loss

        self.optimizer.zero_grad()
        loss.backward()
        for global_param, local_param in zip(self.model.parameters(), local_model.parameters()):
            global_param._grad = local_param.grad
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 40)
        self.optimizer.step()
        self.lr_scheduler.step()
        return loss

    def sample(self, state):
        prob = self.model.pi(state)
        m = Categorical(prob)
        action = m.sample().item()

        return action

    def predict(self, state):
        prob = self.model.pi(state)
        action = prob.argmax()

        return action
