import copy
import torch
import torch.optim as optim
from config import config


class Alg():
    def __init__(self, model):
        self.model = model
        self.target_model = copy.deepcopy(self.model)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.target_model.to(device)
        self.gamma = config['gamma']
        self.lr = config['learning_rate']

        self.mse_loss = torch.nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def predict(self, state):
        with torch.no_grad():
            pred_q = self.model(state).cpu()
        action = pred_q.argmax().item()
        return action

    def learn(self, states, actions, rewards, next_states, done):
        pred_value = self.model(states).gather(1, actions)
        greedy_action = self.model(next_states).max(dim=-1, keepdim=True)[1]
        with torch.no_grad():
            max_v = self.target_model(next_states).gather(1, greedy_action)
            target = rewards + (1 - done) * self.gamma * max_v
        self.optimizer.zero_grad()
        loss = self.mse_loss(pred_value, target)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def update(self):
        self.target_model.load_state_dict(self.model.state_dict())
