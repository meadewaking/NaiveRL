from alg import Alg
from model import Model
from agent import Agent
import gym
from config import config
import torch.multiprocessing as mp


def train(global_agent, rank, global_ep):
    local_model = Model()
    local_alg = Alg(local_model)
    local_agent = Agent(local_alg)
    local_agent.alg.model.load_state_dict(global_agent.alg.model.state_dict())

    env = gym.make(config['env_name'])

    for episode in range(config['max_episode']):
        score = 0
        done = False
        s = env.reset()
        while not done:
            states, actions, rewards = [], [], []
            for t in range(config['sample_batch_steps']):
                # if rank == 0:
                #     env.render()
                a = local_agent.sample(s)
                s_, r, done, info = env.step(a)

                states.append(s)
                actions.append(a)
                rewards.append(r)
                s = s_
                score += r

            global_agent.learn(states, actions, rewards, local_agent.alg.model, s_, done)
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
