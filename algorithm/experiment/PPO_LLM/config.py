config = {

    'env_name': 'Breakout-v4',
    'env_dim': 4,
    'act_dim': 4,

    'max_episode': int(1e4),
    'gamma': 0.99,
    'lambda': 0.95,
    'train_loop': 3,
    'batch': 16,
    'horizon': 32,

    'learning_rate': 1e-6,
    'epsilon_clip': 0.1,
    'entropy_coeff': -0.02,
    'vf_loss_coeff': 0.5,

    'prompts': ['获得高分'],
    'tokenizer_path': 'tinyllama/tokenizer.model',
    'llm_path': 'tinyllama/model.safetensors',
    'max_text_len': 6
}
