import gym
from model import Model
from alg import Alg
from agent import Agent
from config import config


def reset_env(env):
    result = env.reset()
    return result[0] if isinstance(result, tuple) else result


def step_env(env, action):
    result = env.step(action)
    if len(result) == 5:
        state, reward, terminated, truncated, info = result
        return state, reward, terminated or truncated, info
    return result


def main():
    env = gym.make(config['env_name'])
    PG_model = Model()
    PG_alg = Alg(PG_model)
    PG_agent = Agent(PG_alg)

    for episode in range(config['max_episode']):
        score = 0.0
        s = reset_env(env)
        done = False
        states, actions, rewards = [], [], []

        while not done:
            # env.render()
            a = PG_agent.sample(s)
            s_, r, done, info = step_env(env, a)
            states.append(s)
            actions.append(a)
            rewards.append(r)
            s = s_
            score += r

        PG_agent.learn(states, actions, rewards)

        print("episode :{}, score : {}".format(episode, score))


if __name__ == '__main__':
    main()
