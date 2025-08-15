import gym
import numpy as np
import random
import torch
from collections import deque
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.optim as optim

# FIXED hyperparameters

alpha = 0.1
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.9995
min_epsilon = 0.01
num_episodes = 2000
max_steps = 200
memory = deque(maxlen=50000)
lr = 1e-3
batch_size = 64
target_update_freq = 1000

env = gym.make('LunarLander-v2', render_mode="human")

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

print("State Space: ", state_dim)
print("Action Space: ", action_dim)

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        
        # Slightly deeper network
        self.linear_1 = nn.Linear(state_dim, 256)
        self.linear_2 = nn.Linear(256, 128)
        self.linear_3 = nn.Linear(128, 64)
        self.linear_4 = nn.Linear(64, action_dim)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)  # Add some regularization

    def forward(self, x):
        x = self.relu(self.linear_1(x))
        x = self.dropout(x)
        x = self.relu(self.linear_2(x))
        x = self.dropout(x)
        x = self.relu(self.linear_3(x))
        x = self.linear_4(x)
        return x

# Initialize networks
policy_net = DQN(state_dim=state_dim, action_dim=action_dim)
target_net = DQN(state_dim, action_dim)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()  # Set target network to eval mode

optimizer = optim.Adam(policy_net.parameters(), lr=lr)
loss_fn = nn.MSELoss()

def select_action(state, epsilon):
    if random.random() < epsilon:
        return random.randint(0, action_dim - 1)
    else:
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            return policy_net(state).argmax().item()

def train():
    # Wait for enough experiences before training
    if len(memory) < 2000:
        return
    
    batch = random.sample(memory, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = np.array(states, dtype=np.float32)
    next_states = np.array(next_states, dtype=np.float32)

    states = torch.FloatTensor(states)
    actions = torch.LongTensor(actions).unsqueeze(1)
    rewards = torch.FloatTensor(rewards)
    next_states = torch.FloatTensor(next_states)
    dones = torch.FloatTensor(dones)

    # Double DQN improvement
    current_q_values = policy_net(states).gather(1, actions).squeeze(1)
    
    # Use policy network to select actions, target network to evaluate
    next_actions = policy_net(next_states).argmax(1).unsqueeze(1)
    next_q_values = target_net(next_states).gather(1, next_actions).squeeze(1)
    
    target_q_values = rewards + gamma * next_q_values * (1 - dones)

    loss = loss_fn(current_q_values, target_q_values.detach())
    
    optimizer.zero_grad() # Empty or clear the gradients
    loss.backward() # Calculate the gradients
    # Gradient clipping for stability
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
    optimizer.step() # Update the gradients

scores = []
step_count = 0

print(f"Memory buffer size: {memory.maxlen}")
print(f"Will start training after {2000} experiences")

for ep in range(num_episodes):
    reset_result = env.reset()

    if isinstance(reset_result, tuple):
        state, _ = reset_result
    else:
        state = reset_result

    done = False
    score = 0
    episode_steps = 0

    while not done and episode_steps < max_steps:
        action = select_action(state, epsilon)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        state = np.array(state, dtype=np.float32)
        next_state = np.array(next_state, dtype=np.float32)

        memory.append([state, action, reward, next_state, float(done)])
        state = next_state
        score += reward
        step_count += 1
        episode_steps += 1
        
        # Train every step (if enough memory)
        train()
        
        # Update target network based on steps, not episodes
        if step_count % target_update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())
            print(f"Target network updated at step {step_count}")

    scores.append(score)
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

    if ep % 50 == 0:
        avg_score = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
        print(f"Episode {ep}, Score: {score:.2f}, Avg Score: {avg_score:.2f}, Epsilon: {epsilon:.3f}, Memory: {len(memory)}")

print("Training completed!")

# Enhanced plotting
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(scores)
plt.title('Training Scores')
plt.xlabel('Episode')
plt.ylabel('Score')
plt.axhline(y=200, color='r', linestyle='--', label='Solved threshold')
plt.legend()

# Moving average
window_size = 100
if len(scores) >= window_size:
    moving_avg = np.convolve(scores, np.ones(window_size)/window_size, mode='valid')
    plt.subplot(1, 3, 2)
    plt.plot(moving_avg)
    plt.title(f'Moving Average (window={window_size})')
    plt.xlabel('Episode')
    plt.ylabel('Average Score')
    plt.axhline(y=200, color='r', linestyle='--', label='Solved threshold')
    plt.legend()

# Epsilon decay
epsilons = [1.0 * (epsilon_decay ** i) for i in range(len(scores))]
epsilons = [max(min_epsilon, e) for e in epsilons]
plt.subplot(1, 3, 3)
plt.plot(epsilons)
plt.title('Epsilon Decay')
plt.xlabel('Episode')
plt.ylabel('Epsilon')

plt.tight_layout()
plt.show()

final_avg = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
print(f"Final average score: {final_avg:.2f}")
print(f"Best score achieved: {max(scores):.2f}")
print(f"Episodes with positive scores: {sum(1 for s in scores if s > 0)}")