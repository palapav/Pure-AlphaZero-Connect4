# runs the alphazero algorithm 
# alphazero scans the current board state -> tells you best move to go about
# returns a set of training "actual" data in the
# form of a set of tuples at the end of the function
# we also pit the models here at intervals

# in this function: we do the following:
# run a set # of epochs
    # per epoch -> set number of episodes (each episode -> self play game)
            # during each episode per turn -> after set # of MCTS simulations are done -> append state data point
            # end of each episode (turns are over) -> append a terminal data point  update previous
    # at end of epoch -> we train NN model -> return a new NN model with better set of weights
    # we pit the old and current model -> set best model as previous
def run():
    pass