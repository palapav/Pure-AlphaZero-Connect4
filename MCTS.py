import numpy as np
import copy
# MCTS class will call implicit, default constructor that
# resets search tree per game

# define a node in tree

# did not include dirichlet noise (include later)
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
        def __init__(self, board, player_turn, parent, is_terminal,
                     terminal_score, network_prob, value_est, action_taken=None):
            self.board = copy.deepcopy(board)
            self.player_mark = player_turn
            self.parent_node = parent
            self.is_terminal = is_terminal
            self.terminal_score = terminal_score
            self.action_taken = action_taken
            self.children = []

            # vector of possible actions probability to take from current network
            # curr node pi policy
            self.child_priors = network_prob
            self.z_value = value_est

            self.curr_node_wins = 0
            self.curr_node_visits = 0
            self.parent_node_visits = 0

    @staticmethod
    def get_highest_visits(root_children_nodes):
        # don't need a list -> can save on the memory
        children_visits = [child_node.total_visits for child_node in root_children_nodes]
        best_visits_index = children_visits.index(max(children_visits))
        return root_children_nodes[best_visits_index] 

    @staticmethod
    def calculate_ucb_score(curr_node_wins, curr_node_visits, curr_node_prob, parent_node_visits):
        pass
    
    # doesn't need to be instance method
    # is it better to store children UCB scores in every node 
    # or to recalculate every single time?
    # to store -> need to do at end of backprop -> getting calculated regardless
    @staticmethod
    def select_highest_UCB(children_nodes):
        # no numpy here?
        if len(children_nodes) == 0:
            raise ValueError("Needs to have at least one child")
        children_ucb_scores = []
        num_children = len(children_nodes)

        for i in range(num_children):
            child_node = children_nodes[i]
            parent_node = child_node.parent
            curr_node_wins = child_node.total_wins
            curr_node_visits = child_node.total_visits
            curr_node_parent_visits = parent_node.total_visits

            # columns 1 through 7 in connect4 (0 indexed here)
            # played move -> to generate child board b/c root node has no action taken
            # what should action_taken represent?
            played_move = child_node.action_taken
            # need to use parents? -> don't think so
            # initial estimate of taking an action from state s (child node)
            # according to nn policy
            child_move_prob = child_node.child_priors[played_move]


            child_ucb_score = MCTS.calculate_ucb_score(
                                            curr_node_wins, 
                                            curr_node_visits,
                                            child_move_prob,
                                            curr_node_parent_visits
                                            )
            children_ucb_scores.append(child_ucb_score)

            # finding index of max ucb element
            best_child_index = children_ucb_scores.index(max(children_ucb_scores))
            return children_nodes[best_child_index]






    def select(self, root_node):
        curr_node = root_node

        # encounters leaf node (no children)
        if len(curr_node.children) == 0: return curr_node
        else:
            children_nodes = curr_node.children
            # recurses down branch of best_child_node (highest UCB score)
            best_child_node = 

        pass

    def expand(self, leaf_node, alphazero_net):
        pass

    # no more simulations/rollout and backpropagate result of that
    # we now backpropagate value_est from NN or z_score in search tree
    
    def backpropagate(self, leaf_node, alphazero_net):
        pass

    # performs set MCTS simulations
    # we backpropagate 
    # after all set num simulations are done for move
    # we then add (s, pi, maybe z) to data set
    # once find z -> we add z for previous ones
    # consider removing the root node when training?

    # returns a move + stores a tuple in training dataset
    # for one person's turn
    def search(self, alphazero_net, num_simulations, player_mark,
               root_game_board, training_dataset):
        # create the root node here (current state of the board)
        # game board is 42 cell 1D numpy array -> passed from self-play
        # player_mark is the player whose about to place a mark on
        child_priors, value_est = alphazero_net.forward(root_game_board)
        child_priors = child_priors.detach().numpy()
        value_est = value_est.item()
        # can be previous player's action taken -> or only current player?
        root_node = self.Node(root_game_board, player_mark, None, child_priors, value_est)
        for i in range(num_simulations):
            # selection
            leaf_node = self.select(root_node)

            # if leaf node is terminal -> skip expansion -> backprop actual z_value, continue
            # difference between non terminal and terminal leaf nodes
            if leaf_node.is_terminal:
                # this simulated game in MCTS tree has ended -> backprop true score for more
                # accurate results
                self.backpropagate(leaf_node.terminal_score)
                continue

            # expansion -> create new nodes
            self.expand(leaf_node, alphazero_net)
            # backprop -> value estimate
            self.backpropagate(leaf_node, alphazero_net)

        # need to make these functions -> avoid accessing variables?
        # are we returning from pi_policy or from ucb_scores
        # this is wrong -> we will be returning move with highest # of visits
        # this is the stochastic pi policy fed by MCTS refined from initial estimate
        # but initial estimate value itself doesn't improve
        root_node_children = root_node
        best_move = MCTS.get_highest_visits(root_node_children)
        return best_move
        # selected move -> highest action from pi_vector
        # out here we return the selected move & append to our dataset -> done from run_alphazero
