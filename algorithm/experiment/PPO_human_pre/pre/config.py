config = {

    'env_name': 'Breakout-v4',
    'env_dim': 4,
    'act_dim': 4,

    'max_episode': int(3e4),
    'gamma': 0.99,
    'lambda': 0.95,
    'train_loop': 3,
    'sample_batch_steps': 64,

    'learning_rate': 2e-5,
    'epsilon_clip': 0.1,
    'entropy_coeff': -0.01,
    'vf_loss_coeff': 0.5,

    'alpha': 0.1
}
