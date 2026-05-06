# model.py - Neural network that approximates Q-values for each action.
# Diego

import torch
import torch.nn as nn

class DQN(nn.Module):

    def __init__(self, input_dim=12, output_dim=2):
        super().__init__()

        # 12 inputs -> two hidden layers -> 2 outputs (one Q-value per action)
        # 128 neurons per layer is a reasonable starting size for a 12-dim input
        self.net = nn.Sequential(
            nn.Linear(12, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128,2)
        )

    def forward(self, x):
        return self.net(x)
