import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from config import config


class Alg():
    def __init__(self, model):
        self.model = model
        self.learning_rate = config['learning_rate']
        self.gamma = config['gamma']
        self.rho_clip = config['rho_clip']
        self.c_clip = config['c_clip']
        self.entropy_coeff = config['entropy_coeff']
        self.vf_loss_coeff = config['vf_loss_coeff']
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def sample(self, state):
        with torch.no_grad():
            logits = self.model.pi_logits(state.unsqueeze(0))
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return action.item(), log_prob.item()

    def predict(self, state):
        with torch.no_grad():
            logits = self.model.pi_logits(state.unsqueeze(0))
            action = logits.argmax(dim=-1)
        return action.item()

    def compute_vtrace(self, rewards, discounts, values, bootstrap_value, behavior_log_probs, target_log_probs):
        log_rhos = target_log_probs - behavior_log_probs
        rhos = torch.exp(log_rhos)
        clipped_rhos = torch.clamp(rhos, max=self.rho_clip)
        clipped_cs = torch.clamp(rhos, max=self.c_clip)

        vs = torch.zeros_like(values)
        acc = torch.zeros((), device=self.device)
        next_values = torch.cat([values[1:], bootstrap_value.unsqueeze(0)], dim=0)

        for t in reversed(range(values.shape[0])):
            delta = clipped_rhos[t] * (rewards[t] + discounts[t] * next_values[t] - values[t])
            acc = delta + discounts[t] * clipped_cs[t] * acc
            vs[t] = values[t] + acc

        vs_next = torch.cat([vs[1:], bootstrap_value.unsqueeze(0)], dim=0)
        pg_adv = clipped_rhos * (rewards + discounts * vs_next - values)
        return vs, pg_adv

    def _learn_one(self, trajectory):
        states = torch.as_tensor(np.asarray(trajectory['states']), device=self.device, dtype=torch.float)
        actions = torch.as_tensor(np.asarray(trajectory['actions']), device=self.device, dtype=torch.long)
        rewards = torch.as_tensor(np.asarray(trajectory['rewards']), device=self.device, dtype=torch.float)
        dones = torch.as_tensor(np.asarray(trajectory['dones']), device=self.device, dtype=torch.float)
        behavior_log_probs = torch.as_tensor(np.asarray(trajectory['behavior_log_probs']), device=self.device, dtype=torch.float)
        final_state = torch.as_tensor(np.asarray(trajectory['final_state']), device=self.device, dtype=torch.float)
        terminal = trajectory['terminal']

        logits, values = self.model.pi_v(states)
        values = values.squeeze(-1)
        target_log_probs = F.log_softmax(logits, dim=-1).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            bootstrap_value = torch.zeros((), device=self.device) if terminal else self.model.v(final_state.unsqueeze(0)).squeeze()

        discounts = self.gamma * (1.0 - dones)
        v_targets, pg_adv = self.compute_vtrace(
            rewards, discounts, values.detach(), bootstrap_value, behavior_log_probs, target_log_probs.detach()
        )

        dist = Categorical(logits=logits)
        policy_loss = -(target_log_probs * pg_adv.detach()).mean()
        value_loss = 0.5 * (values - v_targets.detach()).pow(2).mean()
        entropy_loss = dist.entropy().mean()
        loss = policy_loss + self.vf_loss_coeff * value_loss - self.entropy_coeff * entropy_loss
        return loss

    def learn(self, trajectories):
        if isinstance(trajectories, dict):
            trajectories = [trajectories]

        total_loss = None
        total_steps = 0
        for _ in range(config['train_loop']):
            step_loss = None
            step_steps = 0
            for trajectory in trajectories:
                loss = self._learn_one(trajectory)
                steps = len(trajectory['states'])
                step_loss = loss * steps if step_loss is None else step_loss + loss * steps
                step_steps += steps
            step_loss = step_loss / max(step_steps, 1)
            total_loss = step_loss if total_loss is None else total_loss + step_loss
            total_steps += 1

        loss = total_loss / max(total_steps, 1)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 40)
        self.optimizer.step()
        return loss
