import random

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
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate,
                                     weight_decay=1e-2)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.old_pi = copy.deepcopy(model)
        self.buffer = []
        self.buffer_size = 3

    def learn_v4(self, txts, imgs, actions, rewards, txt_final, s_final, done):
        # 将当前转换存储到缓冲区
        self.buffer.append((txts, imgs, actions, rewards, txt_final, s_final, done))

        # 如果缓冲区未达到指定大小，则直接返回，不进行学习
        if len(self.buffer) < self.buffer_size:
            return torch.tensor(0)
        else:
            all_txts, all_imgs, all_actions, all_advantage, all_td_target = [], [], [], [], []
            for data in self.buffer:
                txts, imgs, actions, rewards, txt_final, s_final, done = data
                # 如果已经到达终止状态，那么回报为0，否则使用旧的价值网络估计回报
                R = 0.0 if done else self.old_pi.v(txt_final, s_final).item()
                with torch.no_grad():
                    old_values = self.old_pi.v(txts, imgs)
                r_lst = np.clip(rewards, -1, 1)
                temp_values = old_values.cpu().detach().numpy()
                # 计算广义优势估计
                tds = r_lst + self.gamma * np.append(temp_values[1:], [[R]], axis=0) - temp_values
                advantage = signal.lfilter([1.0], [1.0, -self.gamma * self.lam], tds[::-1])[::-1]
                td_target = advantage + temp_values
                advantage = ((advantage - advantage.mean()) / (advantage.std() + 1e-6))
                advantage = torch.tensor(advantage.copy(), dtype=torch.float).to(self.device)
                td_target = torch.tensor(td_target.copy(), dtype=torch.float).to(self.device)
                all_txts.append(txts)
                all_imgs.append(imgs)
                all_actions.append(actions)
                all_advantage.append(advantage)
                all_td_target.append(td_target)
            txts = torch.cat(all_txts, dim=0)
            imgs = torch.cat(all_imgs, dim=0)
            actions = torch.cat(all_actions, dim=0)
            advantage = torch.cat(all_advantage, dim=0)
            td_target = torch.cat(all_td_target, dim=0)
            self.buffer = []
        # 迭代多次更新模型参数
        batch_size = config['batch']  # 小批次大小

        for i in range(max(actions.shape[0] // batch_size, 1) * config['train_loop']):
            # 随机采样
            if actions.shape[0] > batch_size:
                sample_indexes = random.sample(range(actions.shape[0]), batch_size)
                batch_txts = txts[sample_indexes]
                batch_imgs = imgs[sample_indexes]
                batch_actions = actions[sample_indexes]
                batch_advantage = advantage[sample_indexes]
                batch_td_target = td_target[sample_indexes]
            else:
                # 只选择小批次的数据进行训练
                batch_txts = txts
                batch_imgs = imgs
                batch_actions = actions
                batch_advantage = advantage
                batch_td_target = td_target

            old_pi, old_values = self.old_pi.pi_v(batch_txts, batch_imgs)
            pi, values = self.model.pi_v(batch_txts, batch_imgs)

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

            self.optimizer.zero_grad()
            rl_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()

        self.old_pi.load_state_dict(self.model.state_dict())
        return rl_loss

    def sample(self, txt, img):
        prob = self.old_pi.pi(txt, img).cpu()
        m = Categorical(prob)
        action = m.sample().item()
        return action

    def predict(self, txt, img):
        prob = self.old_pi.pi(txt, img).cpu()
        action = prob.argmax()

        return action
