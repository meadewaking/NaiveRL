import torch
import copy
import numpy as np
from torch.distributions import Categorical
from config import config
import torch.optim as optim


class Alg():
    def __init__(self, model):
        self.model = model
        self.learning_rate = config['learning_rate']
        self.gamma = config['gamma']
        self.lam = config['lambda']
        self.entropy_coeff = config['entropy_coeff']
        self.vf_loss_coeff = config['vf_loss_coeff']
        self.epsilon = config['epsilon_clip']
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Optimization: Use pi_v for shared computation
        self.old_model = copy.deepcopy(self.model)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
    
    def predict(self, state):
        """Action prediction (for inference)"""
        state = state.unsqueeze(0).to(self.device)
        pi = self.model.pi(state)
        dist = Categorical(pi)
        action = dist.sample()
        return action.item()
    
    def sample(self, state):
        """Action sampling (for training)"""
        return self.predict(state)
    
    # ============ Optimization 2: Pure PyTorch GAE ===============
    # Avoid scipy's CPU-GPU conversions
    def compute_gae(self, values, rewards, next_value, done):
        """Pure PyTorch GAE implementation"""
        advantages = torch.zeros_like(rewards).to(self.device)
        last_adv = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_val * (1 - done) - values[t]
            advantages[t] = last_adv = delta + self.gamma * self.lam * (1 - done) * last_adv
        
        returns = advantages + values
        return advantages, returns
    
    def learn(self, states, actions, rewards, s_, done):
        """PPO Update"""
        # Convert data to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        s_ = s_.to(self.device)
        
        # Optimization: Batch compute values using pi_v
        with torch.no_grad():
            all_states = torch.cat([states, s_.unsqueeze(0)], dim=0)
            _, all_values = self.old_model.pi_v(all_states)
            values = all_values[:-1].squeeze()
            final_value = all_values[-1].item()
        
        # Optimization: Pure PyTorch GAE
        next_value = torch.tensor(0.0 if done else final_value).to(self.device)
        advantages, returns = self.compute_gae(values, rewards, next_value, done)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Optimization: Compute old log_probs
        with torch.no_grad():
            pi_old, _ = self.old_model.pi_v(states)
            dist_old = Categorical(pi_old)
            old_log_probs = dist_old.log_prob(actions.squeeze())
        
        # Training loop
        for i in range(config['train_loop']):
            # Optimization: Use pi_v for shared forward pass
            pi, values_new = self.model.pi_v(states)
            dist = Categorical(pi)
            
            # PPO Policy Loss
            log_probs = dist.log_prob(actions.squeeze())
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages.detach()
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages.detach()
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value Function Loss (with clip)
            values_old = self.old_model.v(states).detach()
            values_clipped = values_old + (values_new - values_old).clamp(-self.epsilon, self.epsilon)
            v_loss1 = (values_new - returns.detach()).pow(2)
            v_loss2 = (values_clipped - returns.detach()).pow(2)
            value_loss = torch.max(v_loss1, v_loss2).mean()
            
            # Entropy Loss
            entropy_loss = dist.entropy().mean()
            
            # Total Loss
            loss = policy_loss + self.vf_loss_coeff * value_loss - self.entropy_coeff * entropy_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()
        
        # Optimization: Use parameter copy instead of state_dict
        with torch.no_grad():
            for param_old, param_new in zip(self.old_model.parameters(), self.model.parameters()):
                param_old.copy_(param_new)
        
        return loss