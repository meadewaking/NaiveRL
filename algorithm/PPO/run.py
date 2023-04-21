import datetime

import gym
from model import Model
from alg import Alg
from agent import Agent
from config import config


def main():
    env = gym.make(config['env_name'])
    PPO_model = Model()
    PPO_alg = Alg(PPO_model)
    PPO_agent = Agent(PPO_alg)

    for episode in range(config['max_episode']):
        score = 0.0
        s = env.reset()
        done = False

        while not done:
            states, actions, rewards = [], [], []
            for t in range(config['horizon']):
                # env.render()
                a = PPO_agent.sample(s)
                s_, r, done, info = env.step(a)
                states.append(s)
                actions.append(a)
                rewards.append(r)
                s = s_
                score += r
                if done:
                    break

            PPO_agent.learn(states, actions, rewards, s_, done)

        print("episode :{}, score : {}".format(episode, score))


if __name__ == '__main__':
    starttime = datetime.datetime.now()  # 记录开始时间
    main()
    endtime = datetime.datetime.now()  # 记录结束时间
    print((endtime - starttime).seconds)

