config = {

    'env_name': 'CartPole-v0',
    'act_dim': 2,
    'state_dim': 4,

    'actor_num': 4,

    'max_episode': int(3e3),
    'gamma': 0.98,
    'reward_scale': 100,
    'sample_batch': 60,

    'vf_loss_coeff': 0.5,
    'entropy_coeff': -0.001,
    'learning_rate': 3e-4
}
