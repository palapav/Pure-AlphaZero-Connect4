import nn
from game_utils import self_play
from nn_utils import AlphaLoss, prepare_training_data, save_checkpoint, load_checkpoint, has_checkpoint
import torch.optim as optim
import torch
import sys

class Trainer():
    def __init__(self, net=None, optim=None, loss_function=None, train_loader=None):
        self.net = net
        self.optim = optim
        self.loss_function = loss_function
        self.train_loader = train_loader

    def train(self, epochs):
        losses = []
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_steps = 0
            # network is not actually updating after every epoch 
            for training_data in self.train_loader:
                board_states = training_data[0]

                # print the reshaped boards
                labels = training_data[1]
                pi_policy = labels[:,:-1]
                z_value = labels[:,-1]
                self.optim.zero_grad()

                p_vector, value_est = self.net(board_states)

                if len(z_value) != len(value_est): raise ValueError("predicted and actual value have discrepancy")
                loss = self.loss_function(z_value, value_est, p_vector, pi_policy)

                # backpropagate to compute gradients of parameters
                loss.backward()
                self.optim.step()
                epoch_loss += loss.item()
                epoch_steps += 1

            # average loss of epoch
            losses.append(epoch_loss / epoch_steps)
            print("epoch [%d]: loss %.3f" % (epoch+1, losses[-1]))

        return losses



def train_alphazero(num_iters=10, num_episodes=10):
    # save checkpoint!
    # playing around with the learning rate
    learning_rate = 0.1
    net = nn.AlphaZeroNet()
    # if has_checkpoint("a", 9): net = load_checkpoint(net, "a", 9)
    opt = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    # opt = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=0.01)
    loss_function = AlphaLoss()
    training_examples = []
    # includes epoch losses
    training_losses = []

    for i in range(num_iters):
        # all the training examples accumulated for one iteration of alphazero training
        # 10 episodes/self play games per iteration
        for e in range(num_episodes):
            single_game_dataset = self_play(net)
            print(f"Game {e} finished for iteration {i}")
            training_examples += single_game_dataset
            # save every 20 or 50 (s, p, v)
        print(f"Number of training examples so far: {len(training_examples)}")
        print(f"Preparing Training Data for Iteration {i}")
        train_loader = prepare_training_data(training_examples)
        # change it in a way to just pass in / append new data but keep the Trainer obj in memory
        trainer = Trainer(net=net, optim=opt, loss_function=loss_function, train_loader=train_loader)
        print(f"About to Train on Iteration {i}--")
        losses = trainer.train(epochs=10)

        # print(f"Losses for epochs in iteration {i}: {losses}")
        print(f"Avg loss for iteration {i}:{sum(losses) / len(losses)}")

        """pitting here or continuous update? -> should be after every game?"""

        # net = updated_net
        # save_checkpoint(net, "a", i)

        print(f"Finished training for iteration {i}")

def main():
    # 10 iterations
    train_alphazero()

if __name__ == '__main__':
    main()