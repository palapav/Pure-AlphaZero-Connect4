"""
This module is for saving and loading checkpoints during
training of alphazero and also for graphing and generating loss data
"""
import numpy as np
import torch

class AlphaLoss(torch.nn.Module):
    def __init__(self):
        super(AlphaLoss, self).__init__()
    
    # pi_vector and z_value given from training data
    # value est and p vector given from raw NN
    def forward(self, z_value, value_est, p_vector, pi_vector):
        # MSE for value
        value_error = (value_est - z_value) ** 2

        # cross entropy loss for policy
        pi_transpose = np.transpose(pi_vector)
        cross_entropy = pi_transpose * np.log(p_vector)

        """no regularization yet (go back and include) -> involves C param, model weights (norm)"""

        return value_error - cross_entropy

def save_model():
    pass

def load_model():
    pass

def save_checkpoint():
    pass

def load_checkpoint():
    pass

def graph_loss():
    pass

def prepare_dataset(game_dataset):
    pass
