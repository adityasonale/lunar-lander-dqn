from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
from src.config import Config

def plot_training_results(scores, save_dir=None):
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
    epsilons = [1.0 * (Config.EPSILON_DECAY ** i) for i in range(len(scores))]
    epsilons = [max(Config.EPSILON_START, e) for e in epsilons]
    plt.subplot(1, 3, 3)
    plt.plot(epsilons)
    plt.title('Epsilon Decay')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')

    plt.tight_layout()

    # Save the plot if save_path is provided
    if save_dir is not None:
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Generate filename with current timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"training_results_{timestamp}.png"
        save_path = os.path.join(save_dir, filename)
        
        plt.savefig(save_path)
        print(f"Plot saved to: {save_path}")

    plt.show()