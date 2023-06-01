"""
This module is for saving and loading checkpoints during
training of alphazero and also for graphing and generating loss data
"""
import numpy as np
import torch
import datetime
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
        print(f"pi_vector size:\n{pi_vector.size()}")
        print(f"p_vector size:\n{p_vector.size()}")
        pi_vector_transpose = pi_vector.t()
        cross_entropy = torch.matmul(pi_vector_transpose, torch.log(p_vector))
        print(f"cross entropy:\n{cross_entropy}")

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
        # can make this faster -> reduce copying
        labels_tensor = torch.tensor(np.append(train_example[1], train_example[2]))

        return input_tensor, labels_tensor


def prepare_training_data(games_dataset):
    """extremely slow -> consider converting list to single np array before converting to tensor"""
    train_dataset = CustomDataset(games_dataset)
    """ batch size of 32 too much with so little data? """
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    return train_loader

def save_checkpoint(net, iter_num):
    # current_datetime = datetime.datetime.now()
    # current_datetime_str = current_datetime.strftime("%m-%d-%Y %I:%M:%S %p")
    # need to save to particular file
    checkpoint_path = f"checkpoint-iterq{iter_num}"
    torch.save(net.state.dict(), checkpoint_path)

def load_checkpoint(net, iter_num):
    checkpoint_path = f"checkpoint-iterz{iter_num}"
    net.load_state_dict(torch.load(checkpoint_path))
    return net

# save board states (reshape), outputted policy vector, value estimates
# early in training vs later in training

def graph_loss():
    pass