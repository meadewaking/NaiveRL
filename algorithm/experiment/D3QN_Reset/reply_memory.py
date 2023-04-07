import random
from collections import deque
from config import config


class ReplayMemory(object):
    def __init__(self):
        self.max_size = config['memory_size']
        self.buffer = deque()
        self.batch_size = config['batch_size']

    def append(self, experience):
        if len(self.buffer) > self.max_size:
            self.buffer.popleft()
        self.buffer.append(experience)

    def sample_batch(self):
        batch = random.sample(self.buffer, self.batch_size)
        states = [data[0] for data in batch]
        actions = [data[1] for data in batch]
        rewards = [data[2] for data in batch]
        next_states = [data[3] for data in batch]
        dones = [data[4] for data in batch]
        return states, actions, rewards, next_states, dones
