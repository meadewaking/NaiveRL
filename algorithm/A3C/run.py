from alg import Alg
from model import Model
from agent import Agent
import gym
from config import config
import torch.multiprocessing as mp


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


def train(global_agent, rank, global_ep):
    local_model = Model()
    local_alg = Alg(local_model)
    local_agent = Agent(local_alg)
    local_agent.alg.model.load_state_dict(global_agent.alg.model.state_dict())

    env = gym.make(config['env_name'])

    for episode in range(config['max_episode']):
        score = 0
        done = False
        s = reset_env(env)
        while not done:
            states, actions, rewards = [], [], []
            terminal = False
            for t in range(config['sample_batch_steps']):
                # if rank == 0:
                #     env.render()
                a = local_agent.sample(s)
                s_, r, done, terminal, info = step_env(env, a)

                states.append(s)
                actions.append(a)
                rewards.append(r)
                s = s_
                score += r
                if done:
                    break

            global_agent.learn(states, actions, rewards, local_agent.alg.model, s_, terminal)
            local_agent.alg.model.load_state_dict(global_agent.alg.model.state_dict())
        global_ep.value += 1
        print("rank :{}, episode :{}, score : {}".format(rank, episode, score))


if __name__ == '__main__':
    global_model = Model()
    global_model.share_memory()
    global_alg = Alg(global_model)
    global_agent = Agent(global_alg)
    global_ep = mp.Manager().Value('i', 0)

    processes = []
    for rank in range(config['actor_num']):
        p = mp.Process(target=train, args=(global_agent, rank, global_ep,))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
