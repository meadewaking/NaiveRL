config = {

    'env_name': 'Pendulum-v0',
    'act_dim': 1,
    'state_dim': 3,
    'action_scale': 2,

    'max_episode': int(3e3),
    'gamma': 0.98,
    'tau': 0.01,
    'alpha': 0.01,
    'reward_scale': 10,
    'memory_size': int(5e4),
    'batch_size': 32,
    'observation': int(1e1),
    'train_loop': 20,

    'actor_learning_rate': 5e-4,
    'critic_learning_rate': 5e-4,
}
