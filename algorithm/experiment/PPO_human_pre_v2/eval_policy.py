import os
import time
import torch
from agent import Agent
from common import ImageProcess, make_env, reset_env, step_env
from config import config
from model import Model
from ppo_alg import Alg


def main(model_path=None, episodes=5, render=False):
    model_path = config['ppo_model_file'] if model_path is None else model_path
    env = make_env(render_mode='human' if render else None)
    model = Model(pretrained=False)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)
    agent = Agent(Alg(model, teacher_path=None))
    image_process = ImageProcess()

    for episode in range(1, episodes + 1):
        score = 0.0
        s = reset_env(env)
        done = False
        s_shadow = image_process.StackInit(s)
        while not done:
            a = agent.predict(s_shadow)
            s_, r, done, terminal, info = step_env(env, a)
            s_shadow = image_process.StackNext(s_)
            score += r
            if render:
                time.sleep(0.01)
        print("episode :{}, score : {}".format(episode, score))


if __name__ == '__main__':
    main()
