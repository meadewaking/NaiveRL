import torch
import numpy as np
from config import config
from tinyllama.tokenizer import ChatTokenizer


class Agent():
    def __init__(self, algorithm):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.alg = algorithm
        self.prompts = config['prompts']
        self.tokenizer = ChatTokenizer(config['tokenizer_path'])
        self.max_text_len = config['max_text_len']

    def sample(self, state):
        token_ids = [self.tokenizer.encode(i) for i in self.prompts]
        token_ids_torch = torch.full((len(token_ids), self.max_text_len), 0, dtype=torch.long, device=self.device)
        for k, t in enumerate(token_ids):
            token_ids_torch[k, : len(t)] = torch.tensor(t, dtype=torch.long, device=self.device)

        state = np.expand_dims(state, 0)
        state = torch.tensor(state, device=self.device, dtype=torch.float)
        action = self.alg.sample(token_ids_torch, state)
        return action

    def predict(self, state):
        token_ids = [self.tokenizer.encode(i) for i in self.prompts]
        token_ids_torch = torch.full((len(token_ids), self.max_text_len), 0, dtype=torch.long, device=self.device)
        for k, t in enumerate(token_ids):
            token_ids_torch[k, : len(t)] = torch.tensor(t, dtype=torch.long, device=self.device)

        state = np.expand_dims(state, 0)
        state = torch.tensor(state, device=self.device, dtype=torch.float)
        action = self.alg.predict(token_ids_torch, state)
        return action

    def learn(self, states, actions, rewards, s_, done):
        token_ids = [self.tokenizer.encode(i) for i in self.prompts]
        token_ids_torch = torch.full((len(token_ids), self.max_text_len), 0, dtype=torch.long, device=self.device)
        for k, t in enumerate(token_ids):
            token_ids_torch[k, : len(t)] = torch.tensor(t, dtype=torch.long, device=self.device)
        token_ids_torchs = token_ids_torch.repeat(len(states), 1)

        states = np.asarray(states)
        states = torch.tensor(states, device=self.device, dtype=torch.float)
        actions = np.expand_dims(actions, 1)
        actions = torch.tensor(actions, device=self.device, dtype=torch.long)
        rewards = np.expand_dims(rewards, 1)
        s_ = np.expand_dims(s_, 0)
        s_ = torch.tensor(s_, device=self.device, dtype=torch.float)
        loss = self.alg.learn_v4(token_ids_torchs, states, actions, rewards,token_ids_torch, s_, done)
        return loss.item()
