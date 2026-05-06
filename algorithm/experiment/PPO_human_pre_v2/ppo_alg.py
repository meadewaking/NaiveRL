import copy
import os
import torch
import torch.optim as optim
from torch.distributions import Categorical
import torch.nn.functional as F
from config import config
from model import Teacher


class Alg():
    def __init__(self, model, teacher_path=None):
        self.model = model
        self.learning_rate = config['learning_rate']
        self.gamma = config['gamma']
        self.lam = config['lambda']
        self.entropy_coeff = config['entropy_coeff']
        self.vf_loss_coeff = config['vf_loss_coeff']
        self.epsilon = config['epsilon_clip']
        self.max_grad_norm = config['max_grad_norm']
        self.teacher_kl_coeff = config['teacher_kl_coeff']
        self.teacher_kl_decay = config['teacher_kl_decay']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.teacher = None
        teacher_path = config['bc_model_file'] if teacher_path is None else teacher_path
        if teacher_path and os.path.exists(teacher_path):
            self.teacher = Teacher(pretrained=False).to(self.device)
            state = torch.load(teacher_path, map_location=self.device)
            if hasattr(state, 'state_dict'):
                state = state.state_dict()
            self.teacher.load_state_dict(state, strict=False)
            if config.get('init_from_teacher', True):
                self.model.load_state_dict(state, strict=False)
            self.teacher.eval()
            for param in self.teacher.parameters():
                param.requires_grad = False

        self.model.to(self.device)
        self.old_pi = copy.deepcopy(self.model).to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)

    def compute_gae(self, values, rewards, next_value, terminal):
        advantages = torch.zeros_like(rewards, device=self.device)
        last_adv = torch.zeros((), device=self.device)
        for t in reversed(range(rewards.shape[0])):
            if t == rewards.shape[0] - 1:
                next_val = next_value
                next_non_terminal = 0.0 if terminal else 1.0
            else:
                next_val = values[t + 1]
                next_non_terminal = 1.0
            delta = rewards[t] + self.gamma * next_val * next_non_terminal - values[t]
            last_adv = delta + self.gamma * self.lam * next_non_terminal * last_adv
            advantages[t] = last_adv
        return advantages, advantages + values

    def learn(self, states, actions, rewards, s_, terminal):
        states = states.to(self.device)
        actions = actions.to(self.device).view(-1)
        rewards = torch.clamp(rewards.to(self.device).view(-1), -1, 1)
        s_ = s_.to(self.device)

        with torch.no_grad():
            old_logits, old_values = self.old_pi.logits_v(states)
            old_values = old_values.squeeze(-1)
            old_dist = Categorical(logits=old_logits)
            old_log_probs = old_dist.log_prob(actions)
            next_value = torch.zeros((), device=self.device) if terminal else self.old_pi.v(s_.unsqueeze(0)).squeeze()
            advantages, returns = self.compute_gae(old_values, rewards, next_value, terminal)
            if advantages.numel() > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        for _ in range(config['train_loop']):
            logits, values = self.model.logits_v(states)
            values = values.squeeze(-1)
            dist = Categorical(logits=logits)
            log_probs = dist.log_prob(actions)
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages.detach()
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages.detach()
            policy_loss = -torch.min(surr1, surr2).mean()

            values_clipped = old_values + (values - old_values).clamp(-self.epsilon, self.epsilon)
            v_loss1 = (values - returns.detach()).pow(2)
            v_loss2 = (values_clipped - returns.detach()).pow(2)
            value_loss = torch.max(v_loss1, v_loss2).mean()
            entropy_loss = dist.entropy().mean()
            loss = policy_loss + self.vf_loss_coeff * value_loss - self.entropy_coeff * entropy_loss

            if self.teacher is not None and self.teacher_kl_coeff > 0:
                with torch.no_grad():
                    teacher_logits = self.teacher.logits(states)
                    teacher_prob = F.softmax(teacher_logits, dim=-1)
                student_log_prob = F.log_softmax(logits, dim=-1)
                teacher_loss = F.kl_div(student_log_prob, teacher_prob, reduction='batchmean')
                loss = loss + self.teacher_kl_coeff * teacher_loss

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()

        self.old_pi.load_state_dict(self.model.state_dict())
        self.teacher_kl_coeff *= self.teacher_kl_decay
        return loss

    def sample(self, state):
        with torch.no_grad():
            logits = self.old_pi.logits(state.unsqueeze(0))
            dist = Categorical(logits=logits)
            action = dist.sample()
        return action.item()

    def predict(self, state):
        with torch.no_grad():
            logits = self.old_pi.logits(state.unsqueeze(0))
            action = logits.argmax(dim=-1)
        return action.item()
