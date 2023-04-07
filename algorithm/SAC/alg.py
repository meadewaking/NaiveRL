import copy
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from config import config

class Alg():
    def __init__(self, actor, critic):
        self.actor = actor
        self.critic_1 = critic
        self.critic_2 = critic
        self.target_critic_1 = copy.deepcopy(self.critic_1)
        self.target_critic_2 = copy.deepcopy(self.critic_2)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.actor.to(device)
        self.critic_1.to(device)
        self.critic_2.to(device)
        self.target_critic_1.to(device)
        self.target_critic_2.to(device)
        self.gamma = config['gamma']
        self.tau = config['tau']
        self.alpha = config['alpha']
        self.alr = config['actor_learning_rate']
        self.clr = config['critic_learning_rate']

        self.mse_loss = torch.nn.MSELoss()
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.alr)
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=self.clr)
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=self.clr)

    def predict(self, state):
        mu, std = self.actor(state)
        action = torch.tanh(mu)

        return action

    def sample(self, state):
        mu, std = self.actor(state)
        std = F.softplus(std)
        dist = Normal(mu, std)
        prob = dist.rsample()
        log_prob = dist.log_prob(prob)
        action = torch.tanh(prob)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdims=True)

        return action, log_prob

    def learn(self, states, actions, rewards, next_states, done):
        with torch.no_grad():
            next_act, next_log_pi = self.sample(next_states)
            q1_next = self.target_critic_1(next_states, next_act)
            q2_next = self.target_critic_2(next_states, next_act)
            target_Q = torch.min(q1_next, q2_next) - self.alpha * next_log_pi
            target_Q = rewards + self.gamma * (1. - done) * target_Q
        q1 = self.critic_1(states, actions)
        critic_1_loss = self.mse_loss(q1, target_Q)
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()

        q2 = self.critic_1(states, actions)
        critic_2_loss = self.mse_loss(q2, target_Q)
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        acts, log_pi = self.sample(states)
        q1_pi = self.critic_1(states, acts)
        q2_pi = self.critic_2(states, acts)
        min_q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = ((self.alpha * log_pi) - min_q_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return [actor_loss.item(), critic_1_loss.item(), critic_2_loss.item()]

    def soft_update(self):
        for param_target, param in zip(self.target_critic_1.parameters(), self.critic_1.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)
        for param_target, param in zip(self.target_critic_2.parameters(), self.critic_2.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)