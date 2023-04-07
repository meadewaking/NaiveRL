import torch
from torch.distributions import Categorical
from config import config
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from scipy import signal
import torch.nn.functional as F


class Alg():
    def __init__(self, model):
        self.model = model
        self.learning_rate = config['learning_rate']
        self.gamma = config['gamma']
        self.entropy_coeff = config['entropy_coeff']
        self.vf_loss_coeff = config['vf_loss_coeff']
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def learn(self, states, actions, td_target):
        # idx = np.random.choice(states.shape[0], config['sample_batch'], replace=False)
        states = torch.tensor(states, device=self.device, dtype=torch.float)
        actions = torch.tensor(actions, device=self.device, dtype=torch.long)
        td_target = torch.tensor(td_target, device=self.device, dtype=torch.float)
        advantage = td_target - self.model.v(states)

        pi = self.model.pi(states)
        actions_log_probs = torch.log(pi.gather(1, actions) + 1e-8)
        policy_loss = -((actions_log_probs * advantage.detach()).sum())
        value_delta = self.model.v(states) - td_target.detach()
        value_loss = torch.mul(value_delta, value_delta).sum()
        entropy_loss = (-torch.log(pi + 1e-8) * torch.exp(torch.log(pi + 1e-8))).sum()
        loss = policy_loss + self.vf_loss_coeff * value_loss + self.entropy_coeff * entropy_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 40)
        self.optimizer.step()
        return loss

    def sample(self, state):
        prob = self.model.pi(state).cpu()
        m = Categorical(prob)
        action = m.sample().item()

        return action

    def predict(self, state):
        prob = self.model.pi(state).cpu()
        action = prob.argmax()

        return action
