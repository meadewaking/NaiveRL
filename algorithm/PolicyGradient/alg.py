import torch
from torch.distributions import Categorical
from config import config
import torch.optim as optim


class Alg():
    def __init__(self, model):
        self.model = model
        self.learning_rate = config['learning_rate']
        self.gamma = config['gamma']
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def learn(self, states, actions, rewards):
        R = 0
        dis_r = []
        for r in rewards:
            R = r + self.gamma * R
            dis_r.append(R)
        dis_r.reverse()
        dis_r = torch.tensor(dis_r, device=self.device, dtype=torch.float)
        prob = self.model.pi(states)
        log_prob = Categorical(prob).log_prob(actions)
        loss = torch.mean(-(log_prob * dis_r))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def sample(self, state):
        prob = self.model.pi(state).cpu()
        m = Categorical(prob)
        action = m.sample().item()

        return action

    def predict(self, state):
        prob = self.model.pi(state).cpu()
        action = prob.argmax()

        return action
