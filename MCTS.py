import numpy as np
import math
import copy
import mcts_utils
import torch
import NeuralNetwork
import sys
# refactor later -> inbuilt
import random

MAX_CHILDREN = 7

class MCTS():
    class Node():
        def __init__(self, board, player_turn, parent, child_priors, value_est,
                     is_terminal=False, action_taken=None):
            self.board = copy.deepcopy(board)
            # already played on the board by player
            self.player_mark = player_turn
            self.parent = parent
            self.is_terminal = is_terminal
            # represents action taken by current player on next board
            # action is processed during jump and placed on child board
            self.action_taken = action_taken
            self.children = []

            self.child_priors = child_priors # p vector

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

        # z_scores = np.array([child_node.total_z_score for child_node in root_children_nodes])
        # exp_z_scores = total_z_scores / children_visits
        # print(f"exp z scores: {exp_z_scores}")

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
    def calculate_ucb_score(curr_node_z_score, curr_node_visits, curr_node_prob, parent_node_visits):
        # ucb score is never stored
        if curr_node_visits == 0: return float('+inf')
        # exploration parameter
        C = math.sqrt(2)

        exploitation_term = curr_node_z_score / curr_node_visits
        # now accounting for prior (Bayes rule)
        exploration_term = C * curr_node_prob * math.sqrt(math.log(parent_node_visits) / (1 + curr_node_visits))
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

            # invert total z score for P2 for true representation (only backpropagating P1 scores)
            # total z score (never has to be stored for P2 -> we can always just invert in selection)
            
            """IF PARENT NODE PLAYER MARK IS 2 -> WE NEED TO MAKE A JUMP ON A CHILD NODE THAT SELECTS THE BEST BLACK
            MOVE FROM THE GIVEN STATE -> USE THIS PRINCIPLE TO GUIDE SELECTION """

            """REMEMBER WE ARE GUIDING SELECTION -> NOT JUST SIMPLY KEEPING TRACK OF WIN/LOSSES """

            # play out selections -> check to see why other scenarios don't work -> commit MCTS to documentation
            if parent_node.player_mark == 2: curr_node_total_zscore = curr_node_visits - curr_node_total_zscore
            # if child_node.player_mark == 1: curr_node_total_zscore = curr_node_visits - curr_node_total_zscore
            # if root_node_player_turn == 2: curr_node_total_zscore = curr_node_visits - curr_node_total_zscore

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
        if leaf_node.parent is None: MCTS.add_dirichlet_noise(leaf_node)

        # creating new child nodes based on all available actions
        # new_child_mark = mcts_utils.opponent_player_mark(leaf_node.player_mark)
        for i in range(num_child_outcomes):
            new_child_board = copy.deepcopy(leaf_node_board)
            
            # talk to Alex of why the wiki alignment is not working -> problem with selection?
            # new_child_mark = mcts_utils.opponent_player_mark(leaf_node.player_mark)
            mcts_utils.play_move(new_child_board, available_moves[i], leaf_node.player_mark)

            child_priors, value_est = alphazero_net.forward(MCTS.convert_arr(new_child_board))
            child_priors = child_priors.detach().numpy()[0]
            value_est = value_est.item()

            # set unavailable moves child priors to zeroes
            unavailable_moves = mcts_utils.get_illegal_moves(new_child_board)
            child_priors[unavailable_moves] = 0.00000000

            # determines is_terminal attribute and terminal_score (reward)
            # before finally creating new children nodes
            is_finished, reward = mcts_utils.score_game(new_child_board)

            if is_finished: value_est = reward
            # previous player move on board, owned by current player
            new_child_mark = mcts_utils.opponent_player_mark(leaf_node.player_mark)
            new_child_node = self.Node(
                                new_child_board, new_child_mark, leaf_node,
                                child_priors, value_est, is_finished, available_moves[i]
                                )
            
            # appending all possible children outcomes to the best leaf node
            leaf_node.children.append(new_child_node)
        
        # pick a random child leaf node to backpropagate
        return random.choice(leaf_node.children)


    def backpropagate(self, leaf_node):
        # we're backpropagating from L not from C
        curr_node = leaf_node
        
        while curr_node != None:
            curr_node.visits = curr_node.visits + 1
            
            # updating all current nodes with player 1 terminal scores / value estimates (everything always in terms of player 1)
            # then in selection phase -> inverting for player 2
            curr_node.total_z_score += leaf_node.z_value

            curr_node = curr_node.parent

    # do we need to make this a static method?
    @staticmethod
    # reshapes given 42 input connect4 numpy board into 1 x 42 2D tensor
    def convert_arr(root_game_board):
        return torch.FloatTensor(root_game_board).reshape(1,-1)

    def search(self, alphazero_net, num_simulations, player_mark,
               root_game_board, training_dataset):
        # returns 1 x 7 and 1 x 1
        child_priors, value_est = alphazero_net.forward(MCTS.convert_arr(root_game_board))
        # converting from 1x7 2D tensor to (7,) 1D arr
        child_priors = child_priors.detach().numpy()[0]
        value_est = value_est.item()

        # player_mark = mcts_utils.opponent_player_mark(player_mark)
        root_node = self.Node(root_game_board, player_mark, None, child_priors, value_est)

        for i in range(num_simulations):
            # if i == 2: sys.exit(1)
            leaf_node = self.select(root_node)

            if not leaf_node.is_terminal:
                new_child_node = self.expand(leaf_node, alphazero_net)
                leaf_node = new_child_node

            self.backpropagate(leaf_node)

        pi_policy_vector, chosen_actions = MCTS.create_pi_policy(root_node.children)

        # do inversion here at the root node
        exp_z_score = root_node.total_z_score / root_node.visits
        # print(f"root node total z score: {root_node.total_z_score}")
        # print(f"root node visits: {root_node.visits}")
        # print(f"children node visits:\n{children_visits}")
        # print(f"children z scores:\n{z_scores}")
        # if root_node.player_mark == 2: exp_z_score = (1 - exp_z_score)
        # print(f"root node total z score: {exp_z_score}")
        # print(f"visits: {root_node.visits}")

        # if we maintain a 7 element vector throughout -> don't have to do this -> sub None instead for illegals
        root_pi_policy = MCTS.set_illegal_moves(pi_policy_vector, chosen_actions)

        # print(f"training dataset id: {id(training_dataset)}")

        curr_board_state = copy.deepcopy(root_game_board)
        training_dataset.append([curr_board_state, root_pi_policy, exp_z_score])
        # print(f"training dataset id after : {id(training_dataset)}")
        # print(f"MCTS board state:\n{root_game_board.reshape(6, 7)}")
        # print(f"MCTS policy:\n{root_pi_policy}")
        # print(f"MCTS value est:{exp_z_score}")
        
        # print(np.arange(7))
        print(root_pi_policy)
        # default None -> single value returned, p= needed because skipping some parameters after 7
        return np.random.choice(7, p=root_pi_policy)
        # changing to argmax did improve training
        # return np.argmax(root_pi_policy)

    # ucb parameters are not explorative enough -> increase constant
    # beginning of training -> higher exploration
    # print out exploration vs exploitation
    # beginning of training -> exploration term should take over
    # try to set it to high
    
#--------- MCTS search sanity check --------------
def main():
    alphazero_nn = NeuralNetwork.AlphaZeroNet()
    # root node of an empty board is owned by player 1
    # root node makes a move on possible child boards owned by player 2
    # wins/visits owned at root node
    mcts_test_board = np.array([0, 0, 0, 0, 0, 0, 0,
                               1, 1, 2, 1, 2, 2, 1,
                               2, 2, 1, 1, 2, 2, 2,
                               1, 1, 2, 2, 2, 1, 1,
                               2, 2, 1, 1, 1, 2, 1,
                               1, 1, 1, 2, 1, 2, 1])
    
    mcts_test_board2 = np.array([0, 0, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0, 1,
                                 0, 0, 0, 1, 0, 0, 1,
                                 0, 0, 0, 2, 0, 0, 2,
                                 0, 0, 2, 2, 2, 1, 1,
                                 1, 2, 2, 2, 1, 1, 1])
    
    mcts = MCTS()
    root_player_mark = 1
    training_dataset = []
    player_one_move = mcts.search(alphazero_nn, 500, root_player_mark, mcts_test_board2, training_dataset)
    # should be between columns 0 to 6
    print(f"Player one next best move untrained: {player_one_move}")

if __name__ == '__main__':
    main()
