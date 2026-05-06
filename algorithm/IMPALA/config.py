config = {

    'env_name': 'CartPole-v1',
    'env_dim': 4,
    'act_dim': 2,

    'max_episode': int(1e3),
    'actor_num': 4,
    'rollout_len': 64,

    'gamma': 0.99,
    'learning_rate': 3e-4,
    'train_loop': 2,

    'rho_clip': 1.0,
    'c_clip': 1.0,
    'entropy_coeff': 0.01,
    'vf_loss_coeff': 0.5,
}
