import torch


class Agent():
    def __init__(self, algorithm):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.alg = algorithm

    def sample(self, state):
        state = torch.tensor(state, device=self.device, dtype=torch.float)
        action = self.alg.sample(state)
        return action

    def predict(self, state):
        state = torch.tensor(state, device=self.device, dtype=torch.float)
        action = self.alg.predict(state)
        return action

    def learn(self, states, actions, rewards):
        states = torch.tensor(states, device=self.device, dtype=torch.float)
        actions = torch.tensor(actions, device=self.device, dtype=torch.long)

        loss = self.alg.learn(states, actions, rewards)
        return loss.item()
