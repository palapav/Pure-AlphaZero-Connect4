"""
This module is for saving and loading checkpoints during
training of alphazero and also for graphing and generating loss data
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class AlphaLoss(torch.nn.Module):
    def __init__(self):
        super(AlphaLoss, self).__init__()
    
    # pi_vector and z_value given from training data
    # value est and p vector given from raw NN
    # inputs are all tensors
    def forward(self, z_value, value_est, p_vector, pi_vector):
        # MSE for value
        value_error = (value_est - z_value) ** 2

        # cross entropy loss for policy
        cross_entropy = torch.dot(pi_vector, torch.log(p_vector))

        """no regularization yet (go back and include) -> involves C param, model weights (norm)"""

        return value_error - cross_entropy

# creating PyTorch ready dataset for (s, p, v)
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        train_example = self.data[index]
        
        """type compatability here?"""
        # input features at index 0 (board state)
        input_tensor = torch.FloatTensor(train_example[0])
        # policy vector at index 1 and value at index 2
        labels_tensor = torch.tensor([train_example[1], train_example[2]])

        return input_tensor, labels_tensor


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

def prepare_training_data(game_dataset):
    train_dataset = CustomDataset(game_dataset)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    return train_loader
