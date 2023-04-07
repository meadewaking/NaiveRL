config = {

    'env_name': 'CartPole-v0',
    'env_dim': 4,
    'act_dim': 2,

    'max_episode': int(1e3),
    'gamma': 0.99,
    'lambda': 0.95,
    'train_loop': 3,
    'sample_steps': 32,

    'learning_rate': 1e-3,
    'epsilon_clip': 0.1,
    'entropy_coeff': -0.01,
    'vf_loss_coeff': 0.5
}
