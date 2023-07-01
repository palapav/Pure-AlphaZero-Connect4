"""game_utils.py -> to be used to only generate one game of self play"""

import numpy as np
import NeuralNetwork
from mcts import MCTS
import mcts_utils

CONNECT4_GRID = 42
EMPTY = 0
COLUMNS = 7
ROWS = 6

# plays one game between 2 players for one alphazero net in training
# per game -> we run 500 MCTS simulations per turn
# each MCTS simulation will return a move placed on the Game board
# train the model after every game of self play
def self_play(self, alphazero_net):
    game_board = np.zeros(CONNECT4_GRID)
    game_dataset = []
    # need to rename variables
    game_over = False
    reward = None
    # root player -> player 1 on empty board
    root_player_mark = mcts_utils.opponent_player_mark(self.player_two_mark)
    while not game_over:
        # initial move for player 1
        next_best_move = MCTS().search(
        # 500 -> # of MCTS simulations
                                alphazero_net, 500,
                                root_player_mark,
                                game_board,
                                game_dataset
                                )
        played_mark = mcts_utils.opponent_player_mark(root_player_mark)
        mcts_utils.play_move(game_board, next_best_move, played_mark)
        # print(f"board after move:\n{np.reshape(self.board, (6,7))}")
        game_over, reward = mcts_utils.score_game(game_board)
        root_player_mark = played_mark

    # if reward == 0:
    #     game_dataset[-1][-1] = 0
    #     # previous player's board
    #     game_dataset[-2][-1] = 0
    # else:
    #     # player 1/2 places winning move
    #     game_dataset[-1][-1] = 1
    #     # opposing player played move that resulted in winner's move on next turn
    #     game_dataset[-2][-1] = -1

    game_dataset[-1][-1] = reward

    return game_dataset
            

# -------sanity check for self-play Game ---------
def main():
    alphazero_net = NeuralNetwork.AlphaZeroNet()
    self_play_data = self_play(alphazero_net)
    # print(f"One game of self play training data:\n{self_play_data}")

    second_last_ex = self_play_data[-2]
    print(f"Second to last training example:\n{second_last_ex}")
    last_ex = self_play_data[-1]
    print(f"Last training example:\n{last_ex}")


if __name__ == '__main__':
    main()


