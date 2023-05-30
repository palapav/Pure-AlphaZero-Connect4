"""
This module defines helper functions
for playing the Connect4 game

we will include a self-play function during training
and also a pitting function to play the current
neural network against the old one

This could be defined as a class

This class needs to be refactored to account for self-play games
and MCTS simulation games
"""

import numpy as np

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
            # opponent won -> current player -> -1 score in backprop
            # opponent has won or game not finished yet
            return (False, None)


    # plays one game between 2 players for one alphazero net in training
    # per game -> we run 500 MCTS simulations per turn
    # each MCTS simulation will return a move placed on the Game board
    def self_play(self):
        pass

    # deciding which model is better in training
    def pit_models(self):
        pass

#---- Static methods to facilitate MCTS simulations-------
# will have to refactor later into MCTS utils class most likely
    @staticmethod
    def opponent_player_mark(player_mark):
        if player_mark != 1 and player_mark != 2:
            raise ValueError("Received invalid mark on board")
        return 3 - player_mark
    
    @staticmethod
    def opponent_score(player_score):
        if player_score < 0 or player_score > 1:
            raise ValueError("Given scoer for game is out of range")
        return 1 - player_score
    
    @staticmethod
    def get_avail_moves(board):
        unfilled_cols = [move for move in range(COLUMNS) if board[move] == EMPTY]
        return unfilled_cols
    
    @staticmethod
    def get_illegal_moves(board):
        filled_cols = [move for move in range(COLUMNS) if board[move] != EMPTY]
        return filled_cols
    
    # terrible software design right now -> need to refactor tomorrow
    # playing the selected move on the game board for an episode per epoch
    # we train the network after each epoch
    @staticmethod
    def play_move(board, column, player_mark):
        row = max([r for r in range(ROWS) if board[column + (r * COLUMNS)] == EMPTY])
        board[column + (row * COLUMNS)] = player_mark

    @staticmethod
    def is_win(board, column, player_mark):
        inarow = 3
        row =  min([r for r in range(ROWS) if board[column + (r * COLUMNS)] == player_mark])
        
        def count(offset_row, offset_column):
            for i in range(1, inarow + 1):
                r = row + offset_row * i
                c = column + offset_column * i
                if (
                    r < 0
                    or r >= ROWS
                    or c < 0
                    or c >= COLUMNS
                    or board[c + (r * COLUMNS)] != player_mark
                ):
                    return i - 1
            return inarow

        return (
            count(1, 0) >= inarow  # vertical.
            or (count(0, 1) + count(0, -1)) >= inarow  # horizontal.
            or (count(-1, -1) + count(1, 1)) >= inarow  # top left diagonal.
            or (count(-1, 1) + count(1, -1)) >= inarow  # top right diagonal.
        )

    @staticmethod
    def is_tie(board):
        return not(any(mark == EMPTY for mark in board[0: COLUMNS + 1]))
    
    # we score the game prior to player_mark placing mark on board
    @staticmethod
    def score_game(board, column, player_mark):
        if Game.is_win(board, column, player_mark):
            return (True, 1)
        if Game.is_tie(board):
            return (True, 0)
        else:
            # opponent won -> current player -> -1 score in backprop
            # opponent has won or game not finished yet
            return (False, None)


