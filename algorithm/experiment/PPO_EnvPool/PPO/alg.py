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
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.old_pi = copy.deepcopy(model)

    def learn(self, states, actions, rewards, s_, done):
        for i in range(config['train_loop']):
            advantages = []
            td_targets = []
            for idx, d in enumerate(done):
                s_final = torch.unsqueeze(s_[idx], 0)
                if d:
                    R = 0
                else:
                    R = self.old_pi.v(s_final).item()
                values = self.old_pi.v(states[idx]).cpu().detach().numpy()
                r_lst = np.clip(rewards[idx], -1, 1)
                r_lst = np.expand_dims(r_lst, 1)
                tds = r_lst + self.gamma * np.append(values[1:], [[R]], axis=0) - values
                advantage = signal.lfilter([1.0], [1.0, -self.gamma * self.lam], tds[::-1])[::-1]
                td_target = advantage + values
                advantages.append(advantage)
                td_targets.append(td_target)
            advantage = np.vstack(advantages)
            td_target = np.vstack(td_targets)
            advantage = torch.tensor(advantage.copy(), dtype=torch.float).to(self.device)
            td_target = torch.tensor(td_target.copy(), dtype=torch.float).to(self.device)

            state_vs = torch.squeeze(torch.cat(states.split(1, dim=0), dim=1), dim=0)
            action_vs = torch.squeeze(torch.cat(actions.split(1, dim=0), dim=1), dim=0)
            pi_a = self.model.pi(state_vs).gather(1, action_vs)
            old_pi_a = self.old_pi.pi(state_vs).gather(1, action_vs)
            ratio = torch.exp(torch.log(pi_a + 1e-8) - torch.log(old_pi_a + 1e-8))
            surr1 = ratio * advantage.detach()
            surr2 = torch.clamp(ratio, 1 - config['epsilon_clip'], 1 + config['epsilon_clip']) * advantage.detach()
            policy_loss = -torch.min(surr1, surr2).mean()
            value_clip = self.old_pi.v(state_vs) + (self.model.v(state_vs) - self.old_pi.v(state_vs)).clamp(
                -config['epsilon_clip'], config['epsilon_clip'])
            v_loss1 = (self.model.v(state_vs) - td_target.detach()).pow(2)
            v_loss2 = (value_clip - td_target.detach()).pow(2)
            value_loss = torch.max(v_loss1, v_loss2).mean()
            entropy_loss = (-torch.log(self.model.pi(state_vs) + 1e-8) * torch.exp(
                torch.log(self.model.pi(state_vs) + 1e-8))).mean()
            loss = policy_loss + self.vf_loss_coeff * value_loss + self.entropy_coeff * entropy_loss

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 40)
            self.optimizer.step()
        self.old_pi.load_state_dict(self.model.state_dict())
        return loss

    def sample(self, state):
        prob = self.old_pi.pi(state).cpu()
        action = np.array([Categorical(p).sample().item() for p in prob])

        return action

    def predict(self, state):
        prob = self.old_pi.pi(state).cpu()
        action = prob.argmax()

        return action
