
class Trainer():
    def __init__(self, net=None, optim=None, loss_function=None, train_loader=None):
        self.net = net
        self.optim = optim
        self.loss_function = loss_function
        self.train_loader = train_loader


def run_alphazero(num_iters):
    """
    10 iterations, 5 self play games per iteration, 500 MCTS simulations per turn in a self play game
    once self play game is done -> game dataset is created
    game dataset -> PyTorch ready dataset; train neural network on this dataset
    output updated alphazero neural network
    save model in checkpoint
    record loss in terminal and graphically
    redo process

    maybe we can train after every iteration -> 
    """

    for i in range(num_iters):
        pass



    """no pitting models -> just continuous training after every game of self play for now"""

    pass

def main():
    run_alphazero()

if __name__ == '__main__':
    main()