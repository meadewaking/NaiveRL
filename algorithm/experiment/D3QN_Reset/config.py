config = {

    'env_name': 'CartPole-v0',
    'env_dim': 4,
    'act_dim': 4,
    'num_envs': 20,

    'max_episode': int(4e3),
    'gamma': 0.99,
    'memory_size': int(3e5),
    'batch_size': 512,
    'observation': int(1e1),

    'learning_rate': 1e-4,
    'exploration_rate': 1,
    'update_interval': 500,
    'reset_interval': 10 * 10,
    'expr_step': 1e-3,
    'last_expr': 1e-3
}
