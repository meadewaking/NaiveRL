config = {

    # Use ALE/Breakout-v5 for modern Gym/Gymnasium installs. Keep the v4 name
    # as fallback for old verified environments.
    'env_name': 'ALE/Breakout-v5',
    'fallback_env_names': ['Breakout-v4'],
    'act_dim': 4,

    'frame_size': 84,
    'frame_stack': 4,
    'crop_top': 34,
    'crop_bottom': 194,

    'data_dir': 'tools',
    'states_file': 'tools/states_v2.npy',
    'actions_file': 'tools/actions_v2.npy',
    'bc_model_file': 'tools/manual_model_v2.pth',
    'ppo_model_file': 'tools/ppo_human_pre_v2.pth',
    'score_file': 'tools/scores_v2.npy',

    'max_episode': int(3e4),
    'horizon': 128,
    'batch': 64,
    'train_loop': 4,
    'gamma': 0.99,
    'lambda': 0.95,
    'learning_rate': 1e-4,
    'epsilon_clip': 0.1,
    'entropy_coeff': 0.01,
    'vf_loss_coeff': 0.5,
    'max_grad_norm': 0.5,

    'teacher_kl_coeff': 0.1,
    'teacher_kl_decay': 0.9999,
    'init_from_teacher': True,

    # Public pretrained visual backbone. ConvNeXt-Tiny is the default because
    # it is a representative modern ConvNet, uses normalization that is friendly
    # to small RL batches, and is available directly from torchvision.
    # Supported: convnext_tiny, efficientnet_v2_s.
    'backbone_name': 'convnext_tiny',
    'backbone_pretrained': True,
    'backbone_resize': 0,
    'imagenet_normalize': True,
    'freeze_backbone': False,
    'feature_dim': 512,

    'bc_epoch': 50,
    'bc_batch_size': 128,
    'bc_learning_rate': 3e-4,
    'bc_weight_decay': 1e-4,
    'bc_save_every': 10,
}
