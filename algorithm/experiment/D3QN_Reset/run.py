import envpool
from model import Model
from alg import Alg
from agent import Agent
from config import config
from reply_memory import ReplayMemory
import torch
import numpy as np


def main():
    env = envpool.make_gym("Breakout-v5", num_envs=config['num_envs'])
    DQN_model = Model()
    DQN_alg = Alg(DQN_model)
    DQN_agent = Agent(DQN_alg)
    rpm = ReplayMemory()
    scores = []

    for episode in range(1, config['max_episode']):
        score = 0.0
        s = env.reset()
        done = np.asarray([False, False])

        while not done.any():
            # env.render()
            a = DQN_agent.sample(s)
            s_, r, done, info = env.step(a)
            for i in range(config['num_envs']):
                rpm.append([s[i], a[i], r[i], s_[i], done[i]])
            s = s_
            score += r
            if episode > config['observation']:
                for i in range(10):
                    states, actions, rewards, next_states, dones = rpm.sample_batch()
                    DQN_agent.learn(states, actions, rewards, next_states, dones)
                DQN_agent.drop_reset(p=0.1)

        scores.append(score)
        if episode % 10 == 0:
            np.save('tools/scores_reset.npy', np.array(scores))
        if episode % 50 == 0:
            torch.save(DQN_model, 'tools/DQN.pth')
        print("episode :{}, score : {}".format(episode, score))


if __name__ == '__main__':
    main()
