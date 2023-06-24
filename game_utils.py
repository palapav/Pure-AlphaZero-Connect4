"""
This module defines helper functions
for playing the Connect4 game

we will include a self-play function during training
pitting -> eval module

This could be defined as a class

This class needs to be refactored to account for self-play games
and MCTS simulation games
"""

import numpy as np
import NeuralNetwork
from mcts import MCTS

CONNECT4_GRID = 42
EMPTY = 0
COLUMNS = 7
ROWS = 6

class Game():
    def __init__(self):
        # 42 represents Connect4 Game Board
        # game board passed between players
        # stored as root node in game tree
        # stored during duration of Game object
        self.board = np.zeros(CONNECT4_GRID)
        self.player_one_mark = 1
        self.player_two_mark = 2

    # playing the selected move on the game board for an episode per epoch
    # we train the network after each epoch
    def play_move(self, column, player_mark):
        row = max([r for r in range(ROWS) if self.board[column + (r * COLUMNS)] == EMPTY])
        self.board[column + (row * COLUMNS)] = player_mark

    def is_win(self, column, player_mark):

        inarow = 3
        row =  min([r for r in range(ROWS) if self.board[column + (r * COLUMNS)] == player_mark])
        
        def count(offset_row, offset_column):
            for i in range(1, inarow + 1):
                r = row + offset_row * i
                c = column + offset_column * i
                if (
                    r < 0
                    or r >= ROWS
                    or c < 0
                    or c >= COLUMNS
                    or self.board[c + (r * COLUMNS)] != player_mark
                ):
                    return i - 1
            return inarow

        return (
            count(1, 0) >= inarow  # vertical.
            or (count(0, 1) + count(0, -1)) >= inarow  # horizontal.
            or (count(-1, -1) + count(1, 1)) >= inarow  # top left diagonal.
            or (count(-1, 1) + count(1, -1)) >= inarow  # top right diagonal.
        )

    def is_tie(self):
        return not(any(mark == EMPTY for mark in self.board[0: COLUMNS + 1]))
    
    # we score the game prior to player_mark placing mark on board
    def score_game(self, column, player_mark):
        if self.is_win(column, player_mark):
            return (True, 1)
        if self.is_tie():
            return (True, 0)
        else:
            # game not finished yet
            return (False, None)
    
    def opponent_move(self, player_mark):
        if player_mark != 1 and player_mark != 2:
            raise ValueError("Received invalid mark on board")
        return 3 - player_mark


    # plays one game between 2 players for one alphazero net in training
    # per game -> we run 500 MCTS simulations per turn
    # each MCTS simulation will return a move placed on the Game board
    # train the model after every game of self play
    def self_play(self, alphazero_net):
        game_dataset = []
        is_finished = False
        score = None
        # root player -> player 1 on empty board
        root_player_mark = self.opponent_move(self.player_two_mark)
        while not is_finished:
            # initial move for player 1
            next_best_move = MCTS().search(
            # changed from 500 -> 150 simulations
                                    alphazero_net, 500,
                                    root_player_mark,
                                    self.board,
                                    game_dataset
                                    )
            played_mark = self.opponent_move(root_player_mark)
            self.play_move(next_best_move, played_mark)
            # print(f"board after move:\n{np.reshape(self.board, (6,7))}")
            is_finished, score = self.score_game(next_best_move, played_mark)
            root_player_mark = played_mark

        # game is finished
        # only need to add change score for player one loss terminal score
        # later -> change the last 2 terminal states (one for each player)
        if score == 0:
            game_dataset[-1][-1] = 0
            # previous player's board
            game_dataset[-2][-1] = 0
        else:
            # player 1/2 places winning move
            game_dataset[-1][-1] = 1
            # opposing player played move that resulted in winner's move on next turn
            game_dataset[-2][-1] = -1

        # print(f"finished game:\n{np.reshape(self.board, (6, 7))}")
        # print(f"winning player:\n{played_mark}")
        # print(f"winning score:\n{score}")
        return game_dataset
            

# -------sanity check for self-play Game ---------
def main():
    connect4_game = Game()
    alphazero_net = NeuralNetwork.AlphaZeroNet()
    self_play_data = connect4_game.self_play(alphazero_net)
    # print(f"One game of self play training data:\n{self_play_data}")

    second_last_ex = self_play_data[-2]
    print(f"Second to last training example:\n{second_last_ex}")
    last_ex = self_play_data[-1]
    print(f"Last training example:\n{last_ex}")


if __name__ == '__main__':
    main()


