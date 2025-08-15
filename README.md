# Deep Q-Network for LunarLander-v2

A PyTorch implementation of Deep Q-Network (DQN) to solve OpenAI Gym's LunarLander-v2 environment.

## Features
- Double DQN implementation
- Experience replay with large memory buffer
- Target network with periodic updates
- Epsilon-greedy exploration with decay

## Results
- Achieves average score of 12.87
- Best score achieved: 190.20
- Training plots and performance analysis included
- Above results were achieved with 2000 episodes and can be further improved by increasing number of episodes.

## Usage
```bash
python train.py
