import gym
import numpy as np
import envpool
import torch
import os

from model import Model
from alg import Alg
from agent import Agent
from config import config
from atari_util import ImageProcess, ImageProcess_pool


def main():
    # env = gym.vector.make(config['env_name'], num_envs=config['num_envs'])
    env = envpool.make_gym("Breakout-v5", num_envs=config['num_envs'])
    PPO_model = Model()
    if os.path.exists('../tools/ppo.pth'):
        PPO_model.load_state_dict(torch.load('../tools/ppo.pth').state_dict())
    PPO_alg = Alg(PPO_model)
    PPO_agent = Agent(PPO_alg)
    scores = []

    for episode in range(1, config['max_episode']):
        score = 0.0
        s = env.reset()
        done = np.asarray([False, False])

        while not done.any():
            states, actions, rewards = [], [], []
            for t in range(config['sample_batch_steps']):
                # env.render()
                a = PPO_agent.sample(s)
                s_, r, done, info = env.step(a)
                states.append(s)
                actions.append(a)
                rewards.append(r)
                s = s_
                score += r

            PPO_agent.learn(states, actions, rewards, s, done)
        scores.append(score)
        if episode % 10 == 0:
            np.save('../tools/scores.npy', np.array(scores))
        if episode % 50 == 0:
            torch.save(PPO_model, '../tools/ppo.pth')
            test()
        print("episode :{}, score : {}".format(episode, score))


def test():
    # env = gym.make(config['env_name'], render_mode='human')
    env = envpool.make_gym("Breakout-v5", num_envs=1)
    PPO_model = Model()
    if os.path.exists('../tools/ppo.pth'):
        PPO_model.load_state_dict(torch.load('../tools/ppo.pth').state_dict())
    PPO_alg = Alg(PPO_model)
    PPO_agent = Agent(PPO_alg)
    print('=====================test====================')
    for episode in range(3):
        score = 0.0
        # processer = ImageProcess()
        s = env.reset()
        # s_shadow = processer.StackInit(s)
        done = False

        while not done:
            # env.render()
            a = PPO_agent.predict(s)
            a = np.array([a])
            s_, r, done, info = env.step(a)
            # s_shadow = processer.StackNext(s_)
            s = s_
            score += r

        print("episode :{}, score : {}".format(episode, score))
    env.close()
    print('=====================test====================')


def test_or():
    env = gym.make(config['env_name'])
    # env = envpool.make_gym("Breakout-v5", num_envs=1)
    PPO_model = Model()
    if os.path.exists('../tools/ppo.pth'):
        PPO_model.load_state_dict(torch.load('../tools/ppo.pth').state_dict())
    PPO_alg = Alg(PPO_model)
    PPO_agent = Agent(PPO_alg)
    print('=====================test====================')
    for episode in range(3):
        score = 0.0
        processer = ImageProcess()
        s = env.reset()
        s, r, done, info = env.step(1)
        s_shadow = processer.StackInit(s)
        done = False

        while not done:
            env.render()
            # time.sleep(.01)
            a = PPO_agent.predict(s_shadow)
            s_, r, done, info = env.step(a)
            s_shadow = processer.StackNext(s_)
            score += r

        print("episode :{}, score : {}".format(episode, score))
    env.close()
    print('=====================test====================')


if __name__ == '__main__':
    main()
    # test()
    # test_or()
