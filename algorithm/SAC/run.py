import gym
from model import Pi, Q
from alg import Alg
from agent import Agent
from config import config
from reply_memory import ReplayMemory


def main():
    env = gym.make(config['env_name'])
    actor = Pi()
    critic = Q()
    SAC_alg = Alg(actor, critic)
    SAC_agent = Agent(SAC_alg)
    rpm = ReplayMemory()

    for episode in range(config['max_episode']):
        score = 0.0
        s = env.reset()
        done = False

        while not done:
            # env.render()
            a = SAC_agent.sample(s)
            s_, r, done, info = env.step(a)
            rpm.append([s, a, r, s_, done])
            s = s_
            score += r
        if episode > config['observation']:
            for i in range(config['train_loop']):
                states, actions, rewards, next_states, dones = rpm.sample_batch()
                SAC_agent.learn(states, actions, rewards, next_states, dones)
        print("episode :{}, score : {}".format(episode, score))


if __name__ == '__main__':
    main()
