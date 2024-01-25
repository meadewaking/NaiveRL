import gym
import numpy as np
import torch

from model import Model
from alg import Alg
from agent import Agent
from config import config
from atari_util import ImageProcess

def main():
    env = gym.make(config['env_name'])
    PPO_model = Model()
    PPO_alg = Alg(PPO_model)
    PPO_agent = Agent(PPO_alg)
    image_process = ImageProcess()
    scores = []

    for episode in range(1, config['max_episode']):
        score = 0.0
        s = env.reset()
        done = False
        s_shadow = image_process.StackInit(s)

        while not done:
            states, actions, rewards = [], [], []
            for t in range(config['horizon']):
                env.render()
                a = PPO_agent.sample(s_shadow)
                s_, r, done, info = env.step(a)
                states.append(s_shadow)
                actions.append(a)
                rewards.append(r)
                s_shadow = image_process.StackNext(s_)
                score += r
                if done:
                    break

            PPO_agent.learn(states, actions, rewards, s_shadow, done)
        scores.append(score)
        if episode % 10 == 0:
            np.save('tools/llarp.npy', np.array(scores))
        if episode % 100 == 0:
            torch.save(PPO_model, 'tools/ppo.pth')
        print("episode :{}, score : {}".format(episode, score))


if __name__ == '__main__':
    main()
