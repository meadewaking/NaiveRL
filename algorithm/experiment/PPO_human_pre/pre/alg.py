import random

import torch
import copy
import numpy as np
from torch.distributions import Categorical
from config import config
from scipy import signal
import torch.optim as optim
from model import Teacher


class Alg():
    def __init__(self, model):
        self.model = model
        self.teacher = Teacher()
        self.teacher.load_state_dict(torch.load('../tools/manual_model_600').state_dict(), strict=False)
        # self.model.bone.load_state_dict(self.teacher.bone.state_dict(), strict=False)
        self.model.load_state_dict(self.teacher.state_dict(), strict=False)
        self.teacher.eval()
        self.model.bone.eval()
        for param in self.teacher.parameters():  # 锁死梯度，保证不发生参数更新
            param.requires_grad = False
        self.learning_rate = config['learning_rate']
        self.gamma = config['gamma']
        self.lam = config['lambda']
        self.entropy_coeff = config['entropy_coeff']
        self.vf_loss_coeff = config['vf_loss_coeff']
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate,
                                     weight_decay=1e-2)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.teacher.to(self.device)
        self.old_pi = copy.deepcopy(model)
        self.KL = torch.nn.KLDivLoss(reduction='batchmean')
        self.alpha = config['alpha']
        self.buffer = []
        self.buffer_size = 5

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

    def learn_v3(self, states, actions, rewards, s_final, done):
        # 如果已经到达终止状态，那么回报为0，否则使用旧的价值网络估计回报
        R = 0.0 if done else self.old_pi.v(s_final).item()
        with torch.no_grad():
            old_values = self.old_pi.v(states)
        r_lst = np.clip(rewards, -1, 1)
        temp_values = old_values.cpu().detach().numpy()
        # 计算广义优势估计
        tds = r_lst + self.gamma * np.append(temp_values[1:], [[R]], axis=0) - temp_values
        advantage = signal.lfilter([1.0], [1.0, -self.gamma * self.lam], tds[::-1])[::-1]
        td_target = advantage + temp_values
        advantage = torch.tensor(advantage.copy(), dtype=torch.float).to(self.device)
        td_target = torch.tensor(td_target.copy(), dtype=torch.float).to(self.device)

        # 迭代多次更新模型参数
        batch_size = config['batch']  # 小批次大小
        sample_size = config['horizon']  # 采样长度

        for i in range(max(actions.shape[0] // batch_size, 1) * config['train_loop']):
            # 随机采样
            if actions.shape[0] > batch_size:
                sample_indexes = random.sample(range(actions.shape[0]), batch_size)
                batch_states = states[sample_indexes]
                batch_actions = actions[sample_indexes]
                batch_advantage = advantage[sample_indexes]
                batch_td_target = td_target[sample_indexes]
            else:
                # 只选择小批次的数据进行训练
                batch_states = states
                batch_actions = actions
                batch_advantage = advantage
                batch_td_target = td_target

            old_pi, old_values = self.old_pi.pi_v(batch_states)
            pi, values = self.model.pi_v(batch_states)

            # 计算策略比例和损失函数
            pi_a = pi.gather(1, batch_actions)
            old_pi_a = old_pi.gather(1, batch_actions)
            ratio = pi_a / (old_pi_a + 1e-8)
            surr1 = ratio * batch_advantage
            surr2 = torch.clamp(ratio, 1 - config['epsilon_clip'], 1 + config['epsilon_clip']) * batch_advantage
            policy_loss = -torch.min(surr1, surr2).mean()
            # 计算值函数损失
            value_clip = old_values + (values - old_values).clamp(
                -config['epsilon_clip'], config['epsilon_clip'])
            v_loss1 = (values - batch_td_target).pow(2)
            v_loss2 = (value_clip - batch_td_target).pow(2)
            value_loss = torch.max(v_loss1, v_loss2).mean()
            # 计算熵正则化损失
            entropy_loss = (-torch.log(pi + 1e-8) * torch.exp(torch.log(pi + 1e-8))).mean()
            rl_loss = policy_loss + self.vf_loss_coeff * value_loss + self.entropy_coeff * entropy_loss
            # 计算策略蒸馏损失
            student_prob = torch.log_softmax(pi, dim=-1)
            teacher_prob = torch.softmax(self.teacher.pi(batch_states), dim=-1)
            supervise_loss = self.KL(student_prob, teacher_prob.detach())
            # 将总的损失函数和策略蒸馏损失函数加权求和
            loss = rl_loss + self.alpha * supervise_loss

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
            self.optimizer.step()

        self.old_pi.load_state_dict(self.model.state_dict())
        self.alpha *= config['alpha_ed']
        return loss

    def learn_v4(self, states, actions, rewards, s_final, done):
        # 将当前转换存储到缓冲区
        self.buffer.append((states, actions, rewards, s_final, done))

        # 如果缓冲区未达到指定大小，则直接返回，不进行学习
        if len(self.buffer) < self.buffer_size:
            return torch.tensor(0)
        else:
            all_states, all_actions, all_advantage, all_td_target = [], [], [], []
            for data in self.buffer:
                states, actions, rewards, s_final, done = data
                # 如果已经到达终止状态，那么回报为0，否则使用旧的价值网络估计回报
                R = 0.0 if done else self.old_pi.v(s_final).item()
                with torch.no_grad():
                    old_values = self.old_pi.v(states)
                r_lst = np.clip(rewards, -1, 1)
                temp_values = old_values.cpu().detach().numpy()
                # 计算广义优势估计
                tds = r_lst + self.gamma * np.append(temp_values[1:], [[R]], axis=0) - temp_values
                advantage = signal.lfilter([1.0], [1.0, -self.gamma * self.lam], tds[::-1])[::-1]
                td_target = advantage + temp_values
                advantage = ((advantage - advantage.mean()) / (advantage.std() + 1e-6))
                advantage = torch.tensor(advantage.copy(), dtype=torch.float).to(self.device)
                td_target = torch.tensor(td_target.copy(), dtype=torch.float).to(self.device)
                all_states.append(states)
                all_actions.append(actions)
                all_advantage.append(advantage)
                all_td_target.append(td_target)
            states = torch.cat(all_states, dim=0)
            actions = torch.cat(all_actions, dim=0)
            advantage = torch.cat(all_advantage, dim=0)
            td_target = torch.cat(all_td_target, dim=0)
            self.buffer = []
        # 迭代多次更新模型参数
        batch_size = config['batch']  # 小批次大小
        sample_size = config['horizon']  # 采样长度

        for i in range(max(actions.shape[0] // batch_size, 1) * config['train_loop']):
            # 随机采样
            if actions.shape[0] > batch_size:
                sample_indexes = random.sample(range(actions.shape[0]), batch_size)
                batch_states = states[sample_indexes]
                batch_actions = actions[sample_indexes]
                batch_advantage = advantage[sample_indexes]
                batch_td_target = td_target[sample_indexes]
            else:
                # 只选择小批次的数据进行训练
                batch_states = states
                batch_actions = actions
                batch_advantage = advantage
                batch_td_target = td_target

            old_pi, old_values = self.old_pi.pi_v(batch_states)
            pi, values = self.model.pi_v(batch_states)

            # 计算策略比例和损失函数
            pi_a = pi.gather(1, batch_actions)
            old_pi_a = old_pi.gather(1, batch_actions)
            ratio = pi_a / (old_pi_a + 1e-8)
            surr1 = ratio * batch_advantage
            surr2 = torch.clamp(ratio, 1 - config['epsilon_clip'], 1 + config['epsilon_clip']) * batch_advantage
            policy_loss = -torch.min(surr1, surr2).mean()
            # 计算值函数损失
            value_clip = old_values + (values - old_values).clamp(
                -config['epsilon_clip'], config['epsilon_clip'])
            v_loss1 = (values - batch_td_target).pow(2)
            v_loss2 = (value_clip - batch_td_target).pow(2)
            value_loss = torch.max(v_loss1, v_loss2).mean()
            # 计算熵正则化损失
            entropy_loss = (-torch.log(pi + 1e-8) * torch.exp(torch.log(pi + 1e-8))).mean()
            rl_loss = policy_loss + self.vf_loss_coeff * value_loss + self.entropy_coeff * entropy_loss
            # 计算策略蒸馏损失
            student_prob = torch.log_softmax(pi, dim=-1)
            teacher_prob = torch.softmax(self.teacher.pi(batch_states), dim=-1)
            supervise_loss = self.KL(student_prob, teacher_prob.detach())
            # 将总的损失函数和策略蒸馏损失函数加权求和
            loss = rl_loss + self.alpha * supervise_loss

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()

        self.old_pi.load_state_dict(self.model.state_dict())
        self.alpha *= config['alpha_ed']
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
