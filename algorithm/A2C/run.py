from alg import Alg
from model import Model
from agent import Agent
import gym
import copy
from config import config
import torch.multiprocessing as mp


def train(global_model):
    local_model = Model()
    local_alg = Alg(local_model)
    local_alg.device = 'cpu'
    local_alg.model.to('cpu')
    local_agent = Agent(local_alg)
    local_agent.device = 'cpu'
    local_agent.alg.model.load_state_dict(global_model.state_dict())

    env = gym.make(config['env_name'])
    score = 0
    done = False
    s = env.reset()
    states, actions, rewards, mask_lst = [], [], [], []
    while not done:
        a = local_agent.sample(s)
        s_, r, done, info = env.step(a)

        states.append(s)
        actions.append(a)
        rewards.append(r)
        mask_lst.append(1 - done)
        s = s_
        score += r

    return score, states, actions, rewards, mask_lst


if __name__ == '__main__':
    global_model = Model()
    global_model.share_memory()
    global_alg = Alg(global_model)
    global_agent = Agent(global_alg)

    for episode in range(config['max_episode']):
        MP = mp.Pool(config['actor_num'])
        temp = []
        for i in range(config['actor_num']):
            copy_model = copy.deepcopy(global_agent.alg.model)
            temp.append(MP.apply_async(train, args=(copy_model.cpu(),)))
        MP.close()
        MP.join()
        scores, states, actions, rewards, mask_lst = [], [], [], [], []
        for g in temp:
            scores.append(g.get()[0])
            states += (g.get()[1])
            actions += (g.get()[2])
            rewards += (g.get()[3])
            mask_lst += (g.get()[3])
        print("episode :{}, score : {}".format(episode, sum(scores)/config['actor_num']))
        global_agent.learn(states, actions, rewards, mask_lst)
