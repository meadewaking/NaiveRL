import gym
from model import Pi, Q
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
    actor = Pi()
    critic_1 = Q()
    critic_2 = Q()
    SAC_alg = Alg(actor, critic_1, critic_2)
    SAC_agent = Agent(SAC_alg)
    rpm = ReplayMemory()

    for episode in range(config['max_episode']):
        score = 0.0
        s = reset_env(env)
        done = False

        while not done:
            # env.render()
            a = SAC_agent.sample(s)
            s_, r, done, terminal, info = step_env(env, a)
            rpm.append([s, a, r, s_, terminal])
            s = s_
            score += r
        if episode > config['observation']:
            for i in range(config['train_loop']):
                states, actions, rewards, next_states, dones = rpm.sample_batch()
                SAC_agent.learn(states, actions, rewards, next_states, dones)
        print("episode :{}, score : {}".format(episode, score))


if __name__ == '__main__':
    main()
