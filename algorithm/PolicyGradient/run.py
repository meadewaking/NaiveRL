import gym
from model import Model
from alg import Alg
from agent import Agent
from config import config


def main():
    env = gym.make(config['env_name'])
    PG_model = Model()
    PG_alg = Alg(PG_model)
    PG_agent = Agent(PG_alg)

    for episode in range(config['max_episode']):
        score = 0.0
        s = env.reset()
        done = False
        states, actions, rewards = [], [], []

        while not done:
            # env.render()
            a = PG_agent.sample(s)
            s_, r, done, info = env.step(a)
            states.append(s)
            actions.append(a)
            rewards.append(r)
            s = s_
            score += r

        PG_agent.learn(states, actions, rewards)

        print("episode :{}, score : {}".format(episode, score))


if __name__ == '__main__':
    main()
