import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from .dqn import DQN

class DQNAgent:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = DQN(
            config.STATE_DIM, 
            config.ACTION_DIM
        ).to(self.device)

        self.target_net = DQN(
            config.STATE_DIM, 
            config.ACTION_DIM
        ).to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Training components
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.LEARNING_RATE)
        self.loss_fn = nn.MSELoss()
        self.memory = deque(maxlen=config.MEMORY_SIZE)

        # Training state
        self.epsilon = config.EPSILON_START
        self.step_count = 0

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.config.ACTION_DIM - 1)
        else:
            state = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                return self.policy_net(state).argmax().item()
            
    def store_experience(self, state, action, reward, next_state, done):
        self.memory.append([state, action, reward, next_state, done])

    def train_step(self):
        # Wait for enough experiences before training
        if len(self.memory) < 2000:
            return
        
        batch = random.sample(self.memory, self.config.BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = np.array(states, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        # Double DQN improvement
        current_q_values = self.policy_net(states).gather(1, actions).squeeze(1)
        
        # Use policy network to select actions, target network to evaluate
        next_actions = self.policy_net(next_states).argmax(1).unsqueeze(1)
        next_q_values = self.target_net(next_states).gather(1, next_actions).squeeze(1)
        
        target_q_values = rewards + self.config.GAMMA * next_q_values * (1 - dones)

        loss = self.loss_fn(current_q_values, target_q_values.detach())
        
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item()
    
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(
            self.config.EPSILON_END, 
            self.epsilon * self.config.EPSILON_DECAY
        )

    def save_model(self, filepath):
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count
        }, filepath)

    def load_model(self, filepath):
        checkpoint = torch.load(filepath)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.step_count = checkpoint['step_count']