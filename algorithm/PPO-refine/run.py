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
        if isinstance(s, tuple):
            s = s[0]
        done = False

        while not done:
            states, actions, rewards = [], [], []
            for t in range(config['horizon']):
                a = PPO_agent.sample(s)
                result = env.step(a)
                # gym 0.26+ returns 5 values
                if len(result) == 5:
                    s_, r, terminated, truncated, info = result
                    done = terminated or truncated
                else:
                    s_, r, done, info = result
                if isinstance(s_, tuple):
                    s_ = s_[0]
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
    starttime = datetime.datetime.now()
    main()
    endtime = datetime.datetime.now()
    print((endtime - starttime).seconds)