import NeuralNetwork
from game_utils import self_play
from nn_utils import AlphaLoss, prepare_training_data, save_checkpoint, load_checkpoint
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
                # all tensors (actual)
                # print(f"Is grad none @ beginning:{list(self.net.parameters())[0].grad}")
                board_states = training_data[0]
                # print(f"Type for board states set:n{type(board_states)}")
                # print(f"Type of board states single: {type(board_states[0])}")

                # print the reshaped boards
                labels = training_data[1]
                pi_policy = labels[:,:-1]
                # print(f"pi policy:\n{pi_policy}")
                # print(f"Type for policy vect: {type(pi_policy)}")
                z_value = labels[:,-1]
                # print(f"z values from training: {z_value}")
                # print(f"Type of z value: {type(z_value)}")
                # print(f"z_value:\n{z_value}")

                # zero the gradient in the optimizer
                # print(f"Is grad none before zero grad:{list(self.net.parameters())[0].grad}")
                self.optim.zero_grad()

                # print(f"Is grad none after zero grad:{list(self.net.parameters())[0].grad}")

                # print(f"Board states dimensions: {board_states.size()}")
                # print(f"num of z values: {len(z_value)}")

                # get the output of the network (predicted)
                p_vector, value_est = self.net(board_states)
                # print(f"initial p_vector in training:\n{type(p_vector)}")
                # print(f"initial value_est in training:\n{type(value_est)}")

                if len(z_value) != len(value_est): raise ValueError("predicted and actual value have discrepancy")
                # give us a collection of p vectors and value estimates

                # computing loss using loss function
                # print(f"Is grad none before loss:{list(self.net.parameters())[0].grad}")
                loss = self.loss_function(z_value, value_est, p_vector, pi_policy)
                # print(f"Loss type: {type(loss)}")

                # print(f"Is grad none:{list(self.net.parameters())[0].grad}")

                # a = list(self.net.parameters())[0].clone()

                # backpropagate to compute gradients of parameters
                loss.backward()
                # print(f"Is grad none after calling backward:{list(self.net.parameters())[0].grad}")

                # b = list(self.net.parameters())[0].clone()

                # call the optimizer, update model parameters
                self.optim.step()

                # print(f"Did params update: {not torch.equal(a.data, b.data)}")

                epoch_loss += loss.item()
                epoch_steps += 1

            # average loss of epoch
            losses.append(epoch_loss / epoch_steps)
            print("epoch [%d]: loss %.3f" % (epoch+1, losses[-1]))

        return self.net, losses



def train_alphazero(num_iters=10, num_episodes=5):
    # save checkpoint!
    # playing around with the learning rate
    learning_rate = 0.1
    net = NeuralNetwork.AlphaZeroNet()
    # we are using the same optimizer every single time -> that's just using the initial parameters
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
            # check on append operation
            # fixed size replay buffer (make it 100000) (if it ever gets filled up -> drop older policy examples)
            training_examples += single_game_dataset
            # save every 20 or 50 (s, p, v)
        print(f"Number of training examples so far: {len(training_examples)}")
        print(f"Preparing Training Data for Iteration {i}")
        train_loader = prepare_training_data(training_examples)
        opt = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
        # retraining model every time we get a new batch of training data
        trainer = Trainer(net=net, optim=opt, loss_function=loss_function, train_loader=train_loader)
        print(f"About to Train on Iteration {i}--")
        # this is where we are collecting the updated net with new parameters
        # are we actually transferring over parameters?
        updated_net, losses = trainer.train(epochs=10)

        # print(f"Losses for epochs in iteration {i}: {losses}")
        print(f"Avg loss for iteration {i}:{sum(losses) / len(losses)}")

        """pitting here or continuous update? -> should be after every game?"""

        # print(f"model state:\n{net.state_dict()}")

        # old_net = set(old_net.state_dict())
        # print(f"old net:\n{old_net}")
        # new = load_checkpoint(net, 0)
        # new_net = set(net.state_dict())
        # print(f"new net:\n{new_net}")
        # both are having the same parameters
        # if new_net == old_net: raise ValueError("parameters are not updating")

        net = updated_net
        # save_checkpoint(net, i)

        print(f"Finished training for iteration {i}")

def main():
    # 10 iterations
    train_alphazero()

if __name__ == '__main__':
    main()