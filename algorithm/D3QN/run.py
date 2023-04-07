import gym
from model import Model
from alg import Alg
from agent import Agent
from config import config
from reply_memory import ReplayMemory


def main():
    env = gym.make(config['env_name'])
    DQN_model = Model()
    DQN_alg = Alg(DQN_model)
    DQN_agent = Agent(DQN_alg)
    rpm = ReplayMemory()

    for episode in range(config['max_episode']):
        score = 0.0
        s = env.reset()
        done = False

        while not done:
            # env.render()
            a = DQN_agent.sample(s)
            s_, r, done, info = env.step(a)
            rpm.append([s, a, r, s_, done])
            s = s_
            score += r
            if episode > config['observation']:
                states, actions, rewards, next_states, dones = rpm.sample_batch()
                DQN_agent.learn(states, actions, rewards, next_states, dones)

        print("episode :{}, score : {}".format(episode, score))


if __name__ == '__main__':
    main()
