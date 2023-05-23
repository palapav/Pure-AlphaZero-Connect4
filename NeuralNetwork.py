import torch.nn as nn
import numpy as np

# Architecture: multilayered perceptron
class AlphaZeroNet(nn.Module):
    def __init__(self):
        super(AlphaZeroNet, self).__init__()
        # 42 -> 256 -> 256 -> 8 (7 + 1)
        self.fc1 = nn.Linear(in_features=42, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=256)
        self.output = nn.Linear(in_features=256, out_features=8)
        self.relu = nn.ReLU()
        # along rows dimension
        self.softmax = nn.Softmax(dim=0)
    
    # x is the 42 input observation board
    def forward(self, x):
        pass

# ----- unit testing -----
