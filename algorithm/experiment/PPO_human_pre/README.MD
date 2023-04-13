# Breakout VPT-PPO Project
This is an experimental open-source project that utilizes the VPT approach to train a PPO (Proximal Policy Optimization) agent to solve the Breakout environment.

## Overview
The project consists of three main steps:

Collecting interaction data using manual_control.py.
Performing Behavioral Cloning (supervised learning) using supervised_training.py.
Fine-tuning the agent through Reinforcement Learning using pre/run.py.

## Usage
### Step 1: Collect Interaction Data
Run manual_control.py to collect interaction data:

`python manual_control.py`
This script allows you to control the agent manually in the Breakout environment and save the interaction data for supervised training.

### Step 2: Perform Behavioral Cloning (Supervised Learning)
Run supervised_training.py to train the agent using the collected interaction data:

`python supervised_training.py`
The script uses the interaction data to perform Behavioral Cloning, which is a supervised learning process to train the agent to imitate the collected expert behavior.

### Step 3: Fine-tune through Reinforcement Learning
Run pre/run.py to fine-tune the agent using Reinforcement Learning:

`python pre/run.py`
This script fine-tunes the agent using the PPO algorithm, which is an advanced Reinforcement Learning (RL) technique, to improve its performance in the Breakout environment.
