import numpy as np
import copy
# MCTS class will call implicit, default constructor that
# resets search tree per game

# did not include dirichlet noise
class MCTS():

    # Node class defines a Node, which represents
    # the state of the board along with the refined
    # pi policy vector, and value estimate for board
    class Node():
        """
        Root node -> empty board
        board -> the current board prior to the player playing the action
        player_turn -> one who is playing the move on the current board
        pi_policy -> probability over potential child actions
        current node can take -> refined with MCTS (initally)
        """
        def __init__(self, board, player_turn, parent, pi_policy, z_score, ucb_scores):
            pass

    def select(self):
        pass

    def expand(self):
        pass

    # no more simulations/rollout and backpropagate result of that
    # we now backpropagate value_est from NN or z_score in search tree
    
    def backpropagate(self):
        pass

    # performs set MCTS simulations
    # we backpropagate 
    # after all set num simulations are done for move
    # we then add (s, pi, maybe z) to data set
    # once find z -> we add z for previous ones
    # consider removing the root node when training?

    # returns a move + stores a tuple in training dataset
    def search(self, alphazero_net, num_simulations):
        pass