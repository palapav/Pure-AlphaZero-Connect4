import NeuralNetwork
from game_utils import Game
from nn_utils import AlphaLoss, prepare_training_data, save_checkpoint
import torch.optim as optim

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

            for training_data in self.train_loader:
                # all tensors (actual)
                board_states = training_data[0]
                labels = training_data[1]
                print(f"labeled output:\n{labels}")
                pi_policy = labels[:,:-1]
                print(f"pi policy:\n{pi_policy}")
                z_value = labels[:,-1]
                print(f"z_value:\n{z_value}")

                # zero the gradient in the optimizer
                self.optim.zero_grad()

                print(f"Board states dimensions: {board_states.size()}")
                print(f"num of z values: {len(z_value)}")

                # get the output of the network (predicted)
                p_vector, value_est = self.net(board_states)
                print(f"value_est:\n{value_est}")

                if len(z_value) != len(value_est): raise ValueError("predicted and actual value have discrepancy")
                # give us a collection of p vectors and value estimates

                # computing loss using loss function
                loss = self.loss_function(z_value, value_est, p_vector, pi_policy)

                # backpropagate to compute gradients of parameters
                loss.backward()

                # call the optimizer, update model parameters
                self.optim.step()

                epoch_loss += loss.item()
                epoch_steps += 1
            
            # average loss of epoch
            losses.append(epoch_loss / epoch_steps)
            print("epoch [%d]: loss %.3f" % (epoch+1, losses[-1]))
        
        return self.net, losses



def train_alphazero(num_iters=10, num_episodes=2):
    """
    10 iterations, 10 self play games per iteration, 500 MCTS simulations per turn in a self play game
    once self play game is done -> game dataset is created
    game dataset -> PyTorch ready dataset; train neural network on this dataset
    output updated alphazero neural network
    save model in checkpoint
    record loss in terminal and graphically
    redo process

    maybe we can train after every iteration -> 
    """
    """Everything being done on CPU -> not GPU """

    """no pitting models? -> just continuous training after every game of self play for now"""
    """training per iteration and not per game -> not enough data """

    learning_rate = 0.1
    net = NeuralNetwork.AlphaZeroNet()
    opt = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    loss_function = AlphaLoss()

    training_examples = []

    for i in range(num_iters):
        # all the training examples accumulated for one iteration of alphazero training
        # 10 episodes/self play games per iteration
        for e in range(num_episodes):
            single_game_dataset = Game().self_play(net)
            print(f"Game {e} finished for iteration {i}")
            # check on append operation
            # fixed size replay buffer (make it 100000) (if it ever gets filled up -> drop older policy examples)
            training_examples += single_game_dataset
            # save every 20 or 50 (s, p, v)

        print(f"Preparing Training Data for Iteration {i}")
        train_loader = prepare_training_data(training_examples)
        # drop the last batch
        trainer = Trainer(net=net, optim=opt, loss_function=loss_function, train_loader=train_loader)
        print(f"About to Train on Iteration {i}--")
        updated_net, losses = trainer.train(epochs=10)

        print(f"Losses for iteration {i}:\n{losses}")

        """pitting here or continuous update? -> should be after every game?"""
        net = updated_net
        save_checkpoint(net, i)

def main():
    # 10 iterations
    train_alphazero()

if __name__ == '__main__':
    main()