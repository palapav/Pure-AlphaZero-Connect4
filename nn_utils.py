"""
This module is for saving and loading checkpoints during
training of alphazero and also for graphing and generating loss data
"""
import numpy as np
import torch
import datetime
import NeuralNetwork
from torch.utils.data import Dataset, DataLoader
from os.path import exists

class AlphaLoss(torch.nn.Module):
    def __init__(self):
        super(AlphaLoss, self).__init__()

    # pi_vector and z_value given from training data
    # value est and p vector given from raw NN
    # inputs are all tensors

    # pass in example pi vector and p vector
    def forward(self, z_value, value_est, p_vector, pi_vector):
        # MSE for value
        mse_error = (value_est - z_value) ** 2

        # cross entropy loss for policy
        p_vector_transpose = p_vector.t()

        cross_entropy = torch.mm(pi_vector.double(), torch.log(p_vector_transpose).double())

        cross_entropy = torch.diagonal(cross_entropy, 0)
        return (mse_error - cross_entropy).mean()

# creating PyTorch ready dataset for (s, p, v)
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        train_example = self.data[index]
        
        input_tensor = torch.FloatTensor(train_example[0])
        labels_tensor = torch.tensor(np.append(train_example[1], train_example[2]))

        return input_tensor, labels_tensor


def prepare_training_data(games_dataset):
    """extremely slow -> consider converting list to single np array before converting to tensor"""
    train_dataset = CustomDataset(games_dataset)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
    return train_loader

def save_checkpoint(net, letter, iter_num):
    # print(f"Type of net in save checkpoint: {type(net)}")
    # current_datetime = datetime.datetime.now()
    # current_datetime_str = current_datetime.strftime("%m-%d-%Y %I:%M:%S %p")
    # need to save to particular file
    checkpoint_path = f"checkpoints/{letter}/checkpoint-iter{letter}bc{iter_num}"
    torch.save(net.state_dict(), checkpoint_path)

def has_checkpoint(letter, iter_num):
    return exists(f"checkpoints/{letter}/checkpoint-iter{letter}bc{iter_num}")

def load_checkpoint(net, letter, iter_num):
    checkpoint_path = f"checkpoints/{letter}/checkpoint-iter{letter}bc{iter_num}"
    net.load_state_dict(torch.load(checkpoint_path))
    return net

def delete_checkpoints():
    pass

def graph_loss():
    pass

def main():
    net = NeuralNetwork.AlphaZeroNet()
    net = load_checkpoint(net, "a", 9)
    print(f"do we have checkpoint: {has_checkpoint('a', 9)}")
    print(f"net parameters: {net.parameters()}")

if __name__ == "__main__": main()