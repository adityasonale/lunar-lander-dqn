import gym
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from src.agent import DQNAgent
from src.config import Config
from src.utils import plot_training_results

def train_agent(config, save_dir=None):
    # Initialize environment and agent
    env = gym.make(config.ENV_NAME, render_mode=config.RENDER_MODE)
    agent = DQNAgent(config)
    
    scores = []
    losses = []
    
    for episode in range(config.NUM_EPISODES):
        reset_result = env.reset()

        if isinstance(reset_result, tuple):
            state, _ = reset_result
        else:
            state = reset_result

        done = False
        score = 0
        episode_steps = 0
        episode_losses = []

        while not done and episode_steps < config.MAX_STEPS:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            state = np.array(state, dtype=np.float32)
            next_state = np.array(next_state, dtype=np.float32)

            agent.store_experience(state, action, reward, next_state, done)
            state = next_state
            score += reward
            agent.step_count += 1
            episode_steps += 1
            
            # Train every step (if enough memory)
            loss = agent.train_step()
            if loss is not None:
                episode_losses.append(loss)
            
            # Update target network based on steps, not episodes
            if agent.step_count % config.TARGET_UPDATE_FREQ == 0:
                agent.update_target_network()
                print(f"Target network updated at step {agent.step_count}")

        scores.append(score)
        losses.append(np.mean(episode_losses) if episode_losses else 0)
        agent.decay_epsilon()

        if episode % 50 == 0:
            avg_score = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
            print(f"Episode {episode}, Score: {score:.2f}, Avg Score: {avg_score:.2f}, Epsilon: {agent.epsilon:.3f}, Memory: {len(agent.memory)}")

        if save_dir:
            plot_training_results(scores, save_dir)
    
    return scores, losses

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-path', default='plots/', help='Path to save results')
    parser.add_argument('--episodes', type=int, default=2000, help='Number of episodes')
    args = parser.parse_args()
    
    config = Config()
    config.NUM_EPISODES = args.episodes
    
    scores, losses = train_agent(config, args.save_path)
    print(f"Training completed! Average final score: {np.mean(scores[-100:]):.2f}")