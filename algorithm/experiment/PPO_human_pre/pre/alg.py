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
        self.teacher = torch.load('../tools/manual_model_600').eval()
        self.model.bone.load_state_dict(self.teacher.bone.state_dict())
        self.learning_rate = config['learning_rate']
        self.gamma = config['gamma']
        self.lam = config['lambda']
        self.entropy_coeff = config['entropy_coeff']
        self.vf_loss_coeff = config['vf_loss_coeff']
        self.optimizer = optim.AdamW([
            # {'params': self.model.bone.parameters(),
            #  'lr': self.learning_rate / 10},
            {'params': self.model.fc_1.parameters()},
            {'params': self.model.fc_pi.parameters()},
            {'params': self.model.fc_v.parameters()}], lr=self.learning_rate,
            weight_decay=1e-2)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.old_pi = copy.deepcopy(model)
        self.KL = torch.nn.KLDivLoss(reduction='batchmean')
        self.alpha = config['alpha']

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
            rl_loss = policy_loss + self.vf_loss_coeff * value_loss + self.entropy_coeff * entropy_loss

            student_prob = torch.log_softmax(self.model.pi(states), dim=-1)
            teacher_prob = torch.softmax(self.teacher.pi(states), dim=-1)
            supervise_loss = self.KL(student_prob, teacher_prob.detach())

            loss = rl_loss + self.alpha * supervise_loss

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
            self.optimizer.step()
        self.old_pi.load_state_dict(self.model.state_dict())
        self.alpha *= 0.9999
        return loss

    def learn_v2(self, states, actions, rewards, s_, done):
        # 迭代多次更新模型参数
        for i in range(config['train_loop']):
            s_final = torch.unsqueeze(s_, 0)
            # 如果已经到达终止状态，那么回报为0，否则使用旧的价值网络估计回报
            R = 0.0 if done else self.old_pi.v(s_final).item()
            old_pi, old_values = self.old_pi.pi_v(states)
            pi, values = self.model.pi_v(states)
            r_lst = np.clip(rewards, -1, 1)
            temp_values = old_values.cpu().detach().numpy()
            # 计算广义优势估计
            tds = r_lst + self.gamma * np.append(temp_values[1:], [[R]], axis=0) - temp_values
            advantage = signal.lfilter([1.0], [1.0, -self.gamma * self.lam], tds[::-1])[::-1]
            td_target = advantage + temp_values
            advantage = torch.tensor(advantage.copy(), dtype=torch.float).to(self.device)
            td_target = torch.tensor(td_target.copy(), dtype=torch.float).to(self.device)

            # 计算策略比例和损失函数
            pi_a = pi.gather(1, actions)
            old_pi_a = old_pi.gather(1, actions)
            ratio = pi_a / (old_pi_a + 1e-8)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - config['epsilon_clip'], 1 + config['epsilon_clip']) * advantage
            policy_loss = -torch.min(surr1, surr2).mean()
            # 计算值函数损失
            value_clip = old_values + (values - old_values).clamp(
                -config['epsilon_clip'], config['epsilon_clip'])
            v_loss1 = (values - td_target).pow(2)
            v_loss2 = (value_clip - td_target).pow(2)
            value_loss = torch.max(v_loss1, v_loss2).mean()
            # 计算熵正则化损失
            entropy_loss = (-torch.log(pi + 1e-8) * torch.exp(torch.log(pi + 1e-8))).mean()
            rl_loss = policy_loss + self.vf_loss_coeff * value_loss + self.entropy_coeff * entropy_loss
            # 计算策略蒸馏损失
            student_prob = torch.log_softmax(pi, dim=-1)
            teacher_prob = torch.softmax(self.teacher.pi(states), dim=-1)
            supervise_loss = self.KL(student_prob, teacher_prob.detach())
            # 将总的损失函数和策略蒸馏损失函数加权求和
            loss = rl_loss + self.alpha * supervise_loss

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
            self.optimizer.step()
        self.old_pi.load_state_dict(self.model.state_dict())
        self.alpha *= 0.9999
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
