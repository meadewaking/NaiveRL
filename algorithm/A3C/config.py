config = {

    'env_name': 'CartPole-v0',
    'act_dim': 2,
    'state_dim': 4,

    'actor_num': 4,
    'sample_batch_steps': 5,

    'max_episode': int(3e2),
    'gamma': 0.95,
    'reward_scale': 100,

    'vf_loss_coeff': 0.5,
    'entropy_coeff': -0.001,
    'learning_rate': 3e-4,
    'lr_step': 1e5,
    'lr_gamma': 0.9
}
