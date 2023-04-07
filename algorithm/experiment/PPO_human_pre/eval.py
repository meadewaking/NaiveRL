import os
import gym
import torch

from model import Model
from alg import Alg
from agent import Agent
from config import config
from atari_util import ImageProcess

import multiprocessing as mp


def eval(model):
    env = gym.make(config['env_name'])
    PPO_model = Model()
    PPO_model.load_state_dict(torch.load('tools/' + model).state_dict())
    PPO_alg = Alg(PPO_model)
    PPO_agent = Agent(PPO_alg)
    image_process = ImageProcess()
    scores = []

    for episode in range(1, 30):
        score = 0.0
        s = env.reset()
        done = False
        s_shadow = image_process.StackInit(s)

        for step in range(3000):
            # env.render()
            a = PPO_agent.sample(s_shadow)
            s_, r, done, info = env.step(a)
            s_shadow = image_process.StackNext(s_)
            score += r
            if done:
                break
        scores.append(score)
        # print("episode :{}, score : {}".format(episode, score))
    print(model, sum(scores) / len(scores), max(scores), min(scores))


def main():
    models = []
    for model in os.listdir('tools'):
        if 'manual_model_' in model:
            models.append(model)
    print(models)
    processes = []
    for model in models:
        process = mp.Process(target=eval, args=(model,))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()


if __name__ == '__main__':
    main()
