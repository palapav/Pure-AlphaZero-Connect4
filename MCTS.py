import numpy as np
# in built -> math module
# naming modules vs classes -> classes not registering
import math
import copy
import mcts_utils
import torch
import NeuralNetwork
import sys
# MCTS class will call implicit, default constructor that
# resets search tree per game

# we don't have a negative one score in implementation when scoring game -> but take into account when backpropping
class MCTS():
    class Node():
        def __init__(self, board, player_turn, parent, network_prob, value_est,
                     is_terminal=False, terminal_score = None, action_taken=None):
            self.board = copy.deepcopy(board)
            # already played on the board by player
            self.player_mark = player_turn
            self.parent = parent
            self.is_terminal = is_terminal
            # child node stores parent's action terminal score
            self.terminal_score = terminal_score
            # represents action taken by current player on next board
            # action is processed during jump and placed on child board
            self.action_taken = action_taken
            self.children = []

            self.child_priors = network_prob # p vector

            # can remove
            self.z_value = value_est

            self.total_z_score = 0
            self.visits = 0
            self.parent_node_visits = 0

    @staticmethod
    # stochastic pi policy -> refined from initial estimate via MCTS
    def create_pi_policy(root_children_nodes):
        """doesn't have 7 nodes -> indices may be wrong"""
        # need to test ordered children nodes
        if len(root_children_nodes) <= 0:
            raise ValueError("Root node must have at least one child node")

        children_visits = np.array([child_node.visits for child_node in root_children_nodes])
        total_children_visits = np.sum(children_visits)
        # child node z value 
        root_pi_policy = children_visits / total_children_visits

        # root node children z scores
        # child_z_scores = np.array([child_node.z_value for child_node in root_children_nodes])

        child_actions_taken = np.array([child_node.action_taken for child_node in root_children_nodes])
        return root_pi_policy, child_actions_taken
    
    @staticmethod
    def set_illegal_moves(pi_policy_vector, actions):
        if len(pi_policy_vector) != len(actions):
            raise ValueError("Number of child priors do not equal number of available actions")

        root_pi_policy = np.zeros(7)
        root_pi_policy[actions] = pi_policy_vector

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

    # store ucb scores?
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
            curr_node_total_zscore = child_node.total_z_score
            curr_node_visits = child_node.visits
            curr_node_parent_visits = parent_node.visits

            played_move = child_node.action_taken
            """look back at this again -> parent node """
            child_move_prob = parent_node.child_priors[played_move]


            child_ucb_score = MCTS.calculate_ucb_score(
                                            curr_node_total_zscore, 
                                            curr_node_visits,
                                            child_move_prob,
                                            curr_node_parent_visits
                                            )
            children_ucb_scores.append(child_ucb_score)

        # finding index of max ucb element (outside for loop lol)
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
    
    @staticmethod
    def add_dirichlet_noise(leaf_node):
        # select only legal moves
        leaf_node_cp = leaf_node.child_priors
        valid_child_priors = leaf_node_cp[leaf_node_cp != 0.00000000]
        valid_cp_indices = np.arange(len(leaf_node_cp))[leaf_node_cp != 0.00000000]

        # dirichlet distribution
        epsilon = 0.25
        # 96, 128, 192, 256
        valid_child_priors = (1 - epsilon) * np.array(valid_child_priors) + \
        epsilon * np.random.dirichlet(np.zeros([len(valid_child_priors)], dtype=np.float32) + 192)

        leaf_node_cp[valid_cp_indices] = valid_child_priors
        leaf_node.child_priors = leaf_node_cp


    def expand(self, leaf_node, alphazero_net):
        """need to refactor to avoid for loops/maximize numpy"""

        # 42 cell numpy array
        leaf_node_board = leaf_node.board
        available_moves = mcts_utils.get_avail_moves(leaf_node_board)
        num_child_outcomes = len(available_moves)
        if num_child_outcomes == 0:
            raise ValueError("leaf node is full -> should not be here in expansion state")
        
        """dirichlet noise at root node here"""
        if leaf_node.parent is None:
            MCTS.add_dirichlet_noise(leaf_node)
            
        # creating new child nodes based on all available actions
        new_child_mark = mcts_utils.opponent_player_mark(leaf_node.player_mark)
        for i in range(num_child_outcomes):
            new_child_board = copy.deepcopy(leaf_node_board)
            # inserting opponent's mark onto new child board
            # as new nodes own the next board positioning
            mcts_utils.play_move(new_child_board, available_moves[i], new_child_mark)
            # new_child_board[available_moves[i]] = opp_mark

            """insert p vector and value estimate for new child board (with child player move played) """
            child_priors, value_est = alphazero_net.forward(MCTS.convert_arr(new_child_board))
            child_priors = child_priors.detach().numpy()[0]
            value_est = value_est.item()

            # set unavailable moves child priors to zeroes
            unavailable_moves = mcts_utils.get_illegal_moves(new_child_board)
            child_priors[unavailable_moves] = 0.00000000


            # determines is_terminal attribute and terminal_score (reward)
            # before finally creating new children nodes
            is_finished, reward = mcts_utils.score_game(new_child_board, available_moves[i], new_child_mark)
            # refactor to avoid to smaller objects to avoid Node constructor too many parameters
            # Node class owned by MCTS object
            # print(f"My mcts reward:{reward}")
            # necessary for backpropagating terminal states
            if is_finished: value_est = reward
            new_child_node = self.Node(
                                new_child_board, new_child_mark, leaf_node,
                                child_priors, value_est, is_finished, reward, 
                                available_moves[i]
                                )
            
            # appending all possible children outcomes to the best leaf node
            leaf_node.children.append(new_child_node)


    def backpropagate(self, leaf_node):
        curr_node = leaf_node
        
        while curr_node != None:
            curr_node.visits = curr_node.visits + 1

            if curr_node.player_mark == leaf_node.player_mark:
                # we need to backpropagate the terminal scores when available
                # expectation 
                # curr_node.z_value = leaf_node.z_value
                curr_node.total_z_score += leaf_node.z_value
            else:
                # curr_node.z_value = -1 * leaf_node.z_value
                curr_node.total_z_score += (-1 * leaf_node.z_value)

            curr_node = curr_node.parent

    # do we need to make this a static method?
    @staticmethod
    # reshapes given 42 input connect4 numpy board into 1 x 42 2D tensor
    def convert_arr(root_game_board):
        return torch.FloatTensor(root_game_board).reshape(1,-1)

    def search(self, alphazero_net, num_simulations, player_mark,
               root_game_board, training_dataset=None):
        # returns 1 x 7 and 1 x 1
        child_priors, value_est = alphazero_net.forward(MCTS.convert_arr(root_game_board))
        # converting from 1x7 2D tensor to (7,) 1D arr
        child_priors = child_priors.detach().numpy()[0]
        value_est = value_est.item()

        root_node = self.Node(root_game_board, player_mark, None, child_priors, value_est)

        for i in range(num_simulations):
            # selection
            leaf_node = self.select(root_node)

            if leaf_node.is_terminal:
                self.backpropagate(leaf_node)
                continue

            self.expand(leaf_node, alphazero_net)
            self.backpropagate(leaf_node)

        pi_policy_vector, chosen_actions = MCTS.create_pi_policy(root_node.children)
        exp_z_score = root_node.total_z_score / root_node.visits

        # if we maintain a 7 element vector throughout -> don't have to do this -> sub None instead for illegals
        root_pi_policy = MCTS.set_illegal_moves(pi_policy_vector, chosen_actions)

        training_dataset.append([root_game_board, root_pi_policy, exp_z_score])

        return np.argmax(root_pi_policy)
    
#--------- MCTS search sanity check --------------
def main():
    alphazero_nn = NeuralNetwork.AlphaZeroNet()
    # player 2 owns this board (played last move at col 2)
    # player 1 move is root node children
    mcts_test_board = np.array([0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 2,
                                0, 0, 1, 1, 0, 0, 2,
                                0, 1, 1, 1, 0, 0, 2,
                                1, 2, 1, 2, 1, 2, 1])
    
    # incorrect for player 2 -> maybe unable to diff btw 1 and 2
    # next player turn -> 1, so we pass in player 2
    mcts = MCTS()
    root_player_mark = 2
    training_dataset = []
    player_one_move = mcts.search(alphazero_nn, 500, root_player_mark, mcts_test_board, training_dataset)
    # should be between columns 0 to 6
    print(f"Player one next best move untrained: {player_one_move}")

if __name__ == '__main__':
    main()
