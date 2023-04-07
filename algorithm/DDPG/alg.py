import copy
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from config import config
from noise import OrnsteinUhlenbeckNoise

class Alg():
    def __init__(self, actor, critic):
        self.actor = actor
        self.critic = critic
        self.target_actor = copy.deepcopy(actor)
        self.target_critic = copy.deepcopy(critic)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.actor.to(device)
        self.target_actor.to(device)
        self.critic.to(device)
        self.target_critic.to(device)
        self.gamma = config['gamma']
        self.tau = config['tau']
        self.alr = config['actor_learning_rate']
        self.clr = config['critic_learning_rate']
        self.noise = OrnsteinUhlenbeckNoise(mu=np.zeros(1))

        self.mse_loss = torch.nn.MSELoss()
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.alr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.clr)

    def predict(self, state):
        prob = self.actor(state).cpu()
        action = prob.item() + self.noise()[0]

        return [action]

    def learn(self, states, actions, rewards, next_states, done):
        target_Q = rewards + self.gamma * self.target_critic(next_states, self.target_actor(next_states)) * (1 - done)

        current_Q = self.critic(states, actions)
        actor_loss = -self.critic(states, self.actor(states)).mean()
        critic_loss = self.mse_loss(target_Q, current_Q)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        return [actor_loss.item(), critic_loss.item()]

    def soft_update(self):
        for param_target, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)
        for param_target, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)
