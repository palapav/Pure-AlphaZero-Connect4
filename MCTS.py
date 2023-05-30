import numpy as np
# in built -> math module
import math
import copy
import Game
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
        def __init__(self, board, player_turn, parent, network_prob, value_est,
                     is_terminal=False, terminal_score = None, action_taken=None):
            self.board = copy.deepcopy(board)
            self.player_mark = player_turn
            self.parent_node = parent
            self.is_terminal = is_terminal
            self.terminal_score = terminal_score
            # represents action taken by current player on current board/node
            # action is processed once mark of current player is placed on board
            self.action_taken = action_taken
            self.children = []

            # vector of possible actions probability to take from current network
            # curr node pi policy
            # all zero child priors for terminal leaf nodes
            self.child_priors = network_prob # p vector
            # updated to terminal score for terminal nodes
            self.z_value = value_est

            self.curr_node_wins = 0
            self.curr_node_visits = 0
            self.parent_node_visits = 0

    @staticmethod
    # stochastic pi policy -> refined from initial estimate via MCTS
    def create_pi_policy(root_children_nodes):
        # our children nodes need to be ordered via the possible action
        # test here -> see if child_node action takens are in order
        if len(root_children_nodes) != 7:
            raise ValueError("Every node must have 7 children probability")
        # returns pi policy vector (needs to be of length 7 always -> 0s for illegal)
        children_visits = np.array([child_node.curr_node_visits for child_node in root_children_nodes])
        total_children_visits = np.sum(children_visits)
        root_pi_policy = children_visits / total_children_visits
        return root_pi_policy

    @staticmethod
    def calculate_ucb_score(curr_node_wins, curr_node_visits, curr_node_prob, parent_node_visits):
                # ucb score is never stored
        if curr_node_visits == 0: return float('+inf')
        # exploration parameter
        C = math.sqrt(2)

        exploitation_term = curr_node_wins / curr_node_visits
        # now accounting for prior (Bayes rule)
        exploration_term = C * curr_node_prob * math.sqrt(math.log(parent_node_visits) / curr_node_visits)
        return exploitation_term + exploration_term
    
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
            """look back at this again -> parent node """
            child_move_prob = parent_node.child_priors[played_move]


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
            best_child_node = MCTS.select_highest_UCB(children_nodes)
            return self.select(best_child_node)

    def expand(self, leaf_node, alphazero_net):
        """did not add dirichlet noise to root node yet
        need to refactor """

        # 42 cell numpy array
        leaf_node_board = leaf_node.board
        available_moves = Game.get_avail_moves(leaf_node_board)
        num_child_outcomes = len(available_moves)
        if num_child_outcomes == 0:
            raise ValueError("leaf node is full -> should not be here in expansion state")
        unavailable_moves = Game.get_illegal_moves(leaf_node_board)

        for i in range(len(unavailable_moves)):
            leaf_node.child_priors[unavailable_moves[i]] = 0.00000000

            # adding the dirichlet noise
            # if true -> root node
            # if leaf_node.parent is None:
            #     Node.add_dirichlet_noise(leaf_node)
            
            # creating new child nodes based on all available actions
        for i in range(num_child_outcomes):
                new_child_board = leaf_node_board.copy()
                # inserting opponent's mark onto new child board
                # as new nodes own the next board positioning
                play(new_child_board, available_moves[i], opp_mark)
                # new_child_board[available_moves[i]] = opp_mark
                

                # determines is_terminal attribute and terminal_score (reward)
                # before finally creating new children nodes
                is_finished, reward = score_game(new_child_board, available_moves[i], opp_mark)
                is_terminal = False
                if is_finished: is_terminal = True
                new_child_node = Node(new_child_board, opp_mark,
                                      reward, leaf_node, is_terminal, 
                                      available_moves[i])
                
                # appending all possible children outcomes to the best leaf node
                leaf_node.children.append(new_child_node)

    # no more simulations/rollout and backpropagate result of that
    # we now backpropagate value_est from NN or z_score in search tree
    
    def backpropagate(self, leaf_node):
        """z value can represent value_est or terminal_score """
        """current backprop is very bad"""
        """terminal leaf node -> one more move left """
        curr_node = leaf_node
        
        while curr_node != None:
            curr_node.total_visits = curr_node.total_visits + 1

            if curr_node.player_mark == 1:
                if leaf_node.player_mark == 1: curr_node.curr_node_wins += leaf_node.z_value
                else: curr_node.curr_node_wins += (-1 * leaf_node.z_value)
            elif curr_node.player_mark == 2:
                if leaf_node.player_mark == 1: curr_node.curr_node_wins += (-1 * leaf_node.z_value)
                else: curr_node.curr_node_wins += leaf_node.z_value

            curr_node = curr_node.parent



    # performs set MCTS simulations
    # we backpropagate 
    # after all set num simulations are done for move
    # we then add (s, pi, maybe z) to data set
    # once find z -> we add z for previous ones
    # consider removing the root node when training?

    # returns a move + stores a tuple in training dataset
    # for one person's turn
    # player mark -> previously player who played move on root game board
    def search(self, alphazero_net, num_simulations, player_mark,
               root_game_board, training_dataset):
        # create the root node here (current state of the board)
        # game board is 42 cell 1D numpy array -> passed from self-play
        # player_mark is the player whose about to place a mark on
        child_priors, value_est = alphazero_net.forward(root_game_board)
        child_priors = child_priors.detach().numpy()
        value_est = value_est.item()
        # player_mark from previous MCTS search call
        root_node = self.Node(root_game_board, player_mark, None, child_priors, value_est)
        for i in range(num_simulations):
            # selection
            leaf_node = self.select(root_node)

            # if leaf node is terminal -> skip expansion -> backprop actual z_value, continue
            # difference between non terminal and terminal leaf nodes
            if leaf_node.is_terminal:
                # this simulated game in MCTS tree has ended -> backprop true score for more
                # accurate results
                self.backpropagate(leaf_node)
                continue

            # expansion -> create new nodes
            self.expand(leaf_node)
            # backprop -> value estimate
            self.backpropagate(leaf_node)

        # need to make these functions -> avoid accessing variables?
        # are we returning from pi_policy or from ucb_scores
        # this is wrong -> we will be returning move with highest # of visits
        # this is the stochastic pi policy fed by MCTS refined from initial estimate
        # but initial estimate value itself doesn't improve
        root_node_children = root_node
        pi_policy_vector = MCTS.create_pi_policy(root_node_children)
        return np.argmax(pi_policy_vector)
        # selected move -> highest action from pi_vector
        # out here we return the selected move & append to our dataset -> done from run_alphazero
