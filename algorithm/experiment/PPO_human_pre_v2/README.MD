# PPO Human Pre V2

V2 keeps the original three independent stages:

1. `manual_control.py`: collect human trajectories and save `tools/states_v2.npy` / `tools/actions_v2.npy`.
2. `supervised_training.py`: train a public pretrained visual backbone by behavior cloning and save `tools/manual_model_v2.pth`.
3. `ppo_finetune.py`: fine-tune PPO with an optional teacher KL term loaded from the behavior-cloned model.

The stages intentionally stay separate so each artifact can be evaluated before moving on.

Core dependencies are `torch`, `torchvision`, `numpy`, `opencv-python`, `pygame`, and a Gym/ALE Atari install. The current machine can still run compile and synthetic algorithm smoke tests without Atari, but real BC/PPO runs need the Atari stack and torchvision weights available.

Compared with the old experiment:

- replaces VGG16 with a configurable public backbone, defaulting to torchvision `convnext_tiny`;
- keeps the point of the original method: scale the policy/value visual encoder beyond the classic Atari CNN;
- adapts 4 stacked grayscale frames to the 3-channel ImageNet interface with a learnable 1x1 temporal projection;
- stores frames as `uint8` and normalizes inside the model;
- uses logits for cross entropy, PPO distributions, entropy, and teacher KL;
- initializes PPO from the behavior-cloned model by default;
- fine-tunes the whole encoder with a unified PPO learning rate;
- supports Gym 0.26 style reset/step return values;
- keeps all V2 artifacts under `tools/*_v2.*` so old experiment files are not overwritten.

Backbone settings live in `config.py`:

- `backbone_name`: `convnext_tiny` or `efficientnet_v2_s`.
- `backbone_pretrained`: load torchvision ImageNet weights when `True`.
- `backbone_resize`: keep the collected 84x84 frames when `0`; set to `128` or `224` if you want a closer ImageNet input scale at higher compute cost.
- `freeze_backbone`: defaults to `False`; V2 follows the later finding that full-network fine-tuning with one learning rate worked better for this experiment.
