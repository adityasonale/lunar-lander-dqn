import torch.nn as nn

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