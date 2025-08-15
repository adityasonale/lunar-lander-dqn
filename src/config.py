class Config:
    # Environment
    ENV_NAME = 'LunarLander-v2'
    RENDER_MODE = 'human'
    
    # Network Architecture
    STATE_DIM = 8
    ACTION_DIM = 4
    HIDDEN_DIMS = [256, 128, 64]
    
    # Training Hyperparameters
    LEARNING_RATE = 1e-3
    GAMMA = 0.99
    EPSILON_START = 1.0
    EPSILON_END = 0.01
    EPSILON_DECAY = 0.9995
    
    # Memory and Training
    MEMORY_SIZE = 50000
    BATCH_SIZE = 64
    TARGET_UPDATE_FREQ = 1000
    
    # Training Setup
    NUM_EPISODES = 2000
    MAX_STEPS = 200