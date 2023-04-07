import torch
import copy
import numpy as np
from torch.distributions import Categorical
from config import config
from scipy import signal
import torch.optim as optim


class Alg():
    def __init__(self, model):
        self.model = model
        self.learning_rate = config['learning_rate']
        self.gamma = config['gamma']
        self.lam = config['lambda']
        self.entropy_coeff = config['entropy_coeff']
        self.vf_loss_coeff = config['vf_loss_coeff']
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.old_pi = copy.deepcopy(model)

    def learn(self, states, actions, rewards, s_, done):
        for i in range(config['train_loop']):
            s_final = torch.unsqueeze(s_, 0)
            R = 0.0 if done else self.old_pi.v(s_final).item()
            values = self.old_pi.v(states).cpu().detach().numpy()
            r_lst = np.clip(rewards, -1, 1)
            tds = r_lst + self.gamma * np.append(values[1:], [[R]], axis=0) - values
            advantage = signal.lfilter([1.0], [1.0, -self.gamma * self.lam], tds[::-1])[::-1]
            td_target = advantage + values
            advantage = torch.tensor(advantage.copy(), dtype=torch.float).to(self.device)
            td_target = torch.tensor(td_target.copy(), dtype=torch.float).to(self.device)

            pi_a = self.model.pi(states).gather(1, actions)
            old_pi_a = self.old_pi.pi(states).gather(1, actions)
            ratio = torch.exp(torch.log(pi_a + 1e-8) - torch.log(old_pi_a + 1e-8))
            surr1 = ratio * advantage.detach()
            surr2 = torch.clamp(ratio, 1 - config['epsilon_clip'], 1 + config['epsilon_clip']) * advantage.detach()
            policy_loss = -torch.min(surr1, surr2).mean()
            value_clip = self.old_pi.v(states) + (self.model.v(states) - self.old_pi.v(states)).clamp(
                -config['epsilon_clip'], config['epsilon_clip'])
            v_loss1 = (self.model.v(states) - td_target.detach()).pow(2)
            v_loss2 = (value_clip - td_target.detach()).pow(2)
            value_loss = torch.max(v_loss1, v_loss2).mean()
            entropy_loss = (-torch.log(self.model.pi(states) + 1e-8) * torch.exp(
                torch.log(self.model.pi(states) + 1e-8))).mean()
            loss = policy_loss + self.vf_loss_coeff * value_loss + self.entropy_coeff * entropy_loss

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 40)
            self.optimizer.step()
        self.old_pi.load_state_dict(self.model.state_dict())
        return loss

    def sample(self, state):
        prob = self.old_pi.pi(state).cpu()
        m = Categorical(prob)
        action = m.sample().item()

        return action

    def predict(self, state):
        prob = self.old_pi.pi(state).cpu()
        action = prob.argmax()

        return action
