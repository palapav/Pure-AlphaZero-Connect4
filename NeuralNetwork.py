import torch
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
        # tanh here
        self.tanh = nn.Tanh()
    
    # x is the 42 input observation board
    # may be bad practice to convert tensor numpy -> tensor within forward method
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.output(out)

        # splitting up output layer -> softmax first 7 classes (p vector)
        # last 8th class -> value estimate

        """use if/else here depending on if we are getting a 1D or 2D tensor"""
        """later convert everything to 2D tensor -> change later to use view function -> better code"""

        # need to seriously fix this
        # if statement -> 2D tensor (7 x 1, 1x1)
        if out.ndim == 1:
            prob_vector = self.softmax(out[0:7])
            # tanh to this
            value_est= self.tanh(out[7])
        elif out.ndim == 2:
            # could use -1
            prob_vector = self.softmax(out[:,0:7])
            # tanh to this
            value_est= self.tanh(out[:,7])

        return (prob_vector, value_est)

# ----- sanity check unit test -----
def main():
    board = np.array([1, 1, 1, 1, 1, 1, 1,
                    0, 0, 0, 0, 0, 0, 0,
                    1, 1, 1, 1, 1, 1, 1,
                    0, 0, 0, 0, 0, 0, 0,
                    1, 1, 1, 1, 1, 1, 1,
                    0, 0, 0, 0, 0, 0, 0])
    
    alpha_zero = AlphaZeroNet()
    # we will be receiving a numpy 2D board
    tensor_board = torch.FloatTensor(board)
    prob_vector, value = alpha_zero.forward(tensor_board)

    # do the detach/numpy and item separately
    print(f"prob vector: {prob_vector.detach().numpy()}")
    print(f"value estimate: {value.item()}")

if __name__ == "__main__":
    main()
