import gym
import torch
import torch.multiprocessing as mp
from alg import Alg
from model import Model
from agent import Agent
from config import config


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


def collect_trajectory(model_state_dict):
    env = gym.make(config['env_name'])
    local_model = Model()
    local_model.load_state_dict(model_state_dict)
    local_alg = Alg(local_model)
    local_alg.device = torch.device('cpu')
    local_alg.model.to('cpu')
    local_agent = Agent(local_alg)
    local_agent.device = torch.device('cpu')

    score = 0.0
    s = reset_env(env)
    done = False
    states, actions, rewards, dones, behavior_log_probs = [], [], [], [], []

    while not done:
        a, log_prob = local_agent.sample(s)
        s_, r, done, terminal, info = step_env(env, a)

        states.append(s)
        actions.append(a)
        rewards.append(r)
        dones.append(float(terminal))
        behavior_log_probs.append(log_prob)
        s = s_
        score += r

    return {
        'states': states,
        'actions': actions,
        'rewards': rewards,
        'dones': dones,
        'behavior_log_probs': behavior_log_probs,
        'final_state': s,
        'terminal': terminal,
        'score': score,
    }


def main():
    global_model = Model()
    global_alg = Alg(global_model)
    global_agent = Agent(global_alg)

    for episode in range(config['max_episode']):
        model_state = {k: v.detach().cpu() for k, v in global_agent.alg.model.state_dict().items()}
        with mp.Pool(config['actor_num']) as pool:
            jobs = [pool.apply_async(collect_trajectory, args=(model_state,)) for _ in range(config['actor_num'])]
            trajectories = [job.get() for job in jobs]

        scores = [traj['score'] for traj in trajectories]
        loss = global_agent.learn(trajectories)
        print("episode :{}, score : {}, loss : {:.4f}".format(episode, sum(scores) / len(scores), loss))


if __name__ == '__main__':
    main()
