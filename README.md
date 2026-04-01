# naiveRL: A Concise Implementation of Mainstream Reinforcement Learning Algorithms

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-red.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
</p>

**naiveRL** is a concise, educational implementation of mainstream reinforcement learning algorithms.

## Features

- Clean & Readable: ~100 lines per algorithm core
- Unified Structure: model → alg → agent → run
- PyTorch Native: Pure PyTorch implementations
- Benchmark Environments: CartPole, Pendulum, Atari

## Implemented Algorithms

| Algorithm | Type |
|-----------|------|
| A2C | Policy Gradient |
| A3C | Asynchronous A2C |
| PPO | Proximal Policy Optimization |
| **PPO-refine** | Optimized PPO (new!) |
| PPO_human_pre | PPO with VGG16 |
| D3QN | Double Dueling DQN |
| DDPG | Deep Deterministic PG |
| SAC | Soft Actor-Critic |

## Quick Start

```bash
git clone https://github.com/meadewaking/NaiveRL.git
cd NaiveRL/algorithm/PPO
python run.py
```

## PPO-refine Optimizations

- Shared forward pass via `pi_v()` method
- Pure PyTorch GAE (no scipy)
- Vectorized batch operations

**Performance**: ~20% speedup on CPU

## License

MIT