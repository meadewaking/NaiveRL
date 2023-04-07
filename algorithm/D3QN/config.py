config = {

    'env_name': 'CartPole-v0',
    'env_dim': 4,
    'act_dim': 2,

    'max_episode': int(1e3),
    'gamma': 0.99,
    'memory_size': int(1e4),
    'batch_size': 16,
    'observation': int(1e1),

    'learning_rate': 1e-3,
    'exploration_rate': 1,
    'update_interval': 200,
    'expr_step': 1e-3,
    'last_expr': 1e-2
}
