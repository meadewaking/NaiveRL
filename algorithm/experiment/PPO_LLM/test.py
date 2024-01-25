import gym
import torch
import os

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
    if os.path.exists('../tools/ppo.pth'):
        PPO_agent.alg.old_pi.load_state_dict(torch.load('../tools/ppo.pth').state_dict())
    image_process = ImageProcess()
    scores = []

    for episode in range(1, 5):
        score = 0.0
        s = env.reset()
        done = False
        s_shadow = image_process.StackInit(s)

        while not done:
            env.render()
            a = PPO_agent.predict(s_shadow)
            s_, r, done, info = env.step(a)
            s_shadow = image_process.StackNext(s_)
            score += r
        scores.append(score)
        print("episode :{}, score : {}".format(episode, score))


if __name__ == '__main__':
    main()
