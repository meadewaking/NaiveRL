import gym
from model import Model
from alg import Alg
from agent import Agent
from config import config
from reply_memory import ReplayMemory


def reset_env(env):
    result = env.reset()
    return result[0] if isinstance(result, tuple) else result


def step_env(env, action):
    result = env.step(action)
    if len(result) == 5:
        state, reward, terminated, truncated, info = result
        return state, reward, terminated or truncated, terminated, info
    state, reward, done, info = result
    return state, reward, done, done, info


def main():
    env = gym.make(config['env_name'])
    DQN_model = Model()
    DQN_alg = Alg(DQN_model)
    DQN_agent = Agent(DQN_alg)
    rpm = ReplayMemory()

    for episode in range(config['max_episode']):
        score = 0.0
        s = reset_env(env)
        done = False

        while not done:
            # env.render()
            a = DQN_agent.sample(s)
            s_, r, done, terminal, info = step_env(env, a)
            rpm.append([s, a, r, s_, terminal])
            s = s_
            score += r
            if episode > config['observation']:
                states, actions, rewards, next_states, dones = rpm.sample_batch()
                DQN_agent.learn(states, actions, rewards, next_states, dones)

        print("episode :{}, score : {}".format(episode, score))


if __name__ == '__main__':
    main()
