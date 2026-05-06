import os
import numpy as np
import torch
from agent import Agent
from common import ImageProcess, ensure_data_dir, make_env, reset_env, step_env
from config import config
from model import Model
from ppo_alg import Alg


def main():
    ensure_data_dir()
    env = make_env()
    has_checkpoint = os.path.exists(config['bc_model_file']) or os.path.exists(config['ppo_model_file'])
    model = Model(pretrained=not has_checkpoint)
    if os.path.exists(config['ppo_model_file']):
        model.load_state_dict(torch.load(config['ppo_model_file'], map_location='cpu'), strict=False)
    alg = Alg(model, teacher_path=config['bc_model_file'])
    agent = Agent(alg)
    image_process = ImageProcess()
    scores = []

    for episode in range(1, config['max_episode']):
        score = 0.0
        s = reset_env(env)
        done = False
        terminal = False
        s_shadow = image_process.StackInit(s)

        while not done:
            states, actions, rewards = [], [], []
            for _ in range(config['horizon']):
                a = agent.sample(s_shadow)
                s_, r, done, terminal, info = step_env(env, a)
                states.append(s_shadow.copy())
                actions.append(a)
                rewards.append(r)
                s_shadow = image_process.StackNext(s_)
                score += r
                if done:
                    break

            loss = agent.learn(states, actions, rewards, s_shadow, terminal)

        scores.append(score)
        if episode % 10 == 0:
            np.save(config['score_file'], np.asarray(scores))
        if episode % 100 == 0:
            torch.save(agent.alg.model.state_dict(), config['ppo_model_file'])
        print("episode :{}, score : {}, loss : {:.4f}, teacher_kl : {:.6f}".format(
            episode, score, loss, agent.alg.teacher_kl_coeff
        ))


if __name__ == '__main__':
    main()
