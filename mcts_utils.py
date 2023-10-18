import numpy as np
import NeuralNetwork
from scipy.signal import convolve2d

# assign to different module -> constants.py
CONNECT4_GRID = 42
EMPTY = 0
COLUMNS = 7
ROWS = 6

# for scoring optimized wins
HORIZONTAL_KERNEL = np.array([[ 1, 1, 1, 1]])
VERTICAL_KERNEL = np.transpose(HORIZONTAL_KERNEL)
DIAG1_KERNEL = np.eye(4, dtype=np.uint8)
DIAG2_KERNEL = np.fliplr(DIAG1_KERNEL)
detection_kernels = [HORIZONTAL_KERNEL, VERTICAL_KERNEL, DIAG1_KERNEL, DIAG2_KERNEL]

#---- methods to facilitate MCTS simulations-------
def opponent_player_mark(player_mark):
    if player_mark != 1 and player_mark != 2:
        raise ValueError("Received invalid mark on board")
    return 3 - player_mark
    

def opponent_score(player_score):
    if player_score < 0 or player_score > 1:
        raise ValueError("Given scoer for game is out of range")
    return 1 - player_score


def get_avail_moves(board):
    unfilled_cols = [move for move in range(COLUMNS) if board[move] == EMPTY]
    return unfilled_cols


def get_illegal_moves(board):
    filled_cols = [move for move in range(COLUMNS) if board[move] != EMPTY]
    return filled_cols
    
# playing the selected move on the game board for an episode per epoch
# we train the network after each epoch
def play_move(board, column, player_mark):
    row = max([r for r in range(ROWS) if board[column + (r * COLUMNS)] == EMPTY])
    board[column + (row * COLUMNS)] = player_mark

def is_tie(board):
    return not(any(mark == EMPTY for mark in board[0: COLUMNS + 1]))

def check_win(board, player_mark):
    # to use in convolve2D for optimizing connect4 wins -> we should change it to convolve1D soon
    board_2d = np.reshape(board, (6, 7))
    for kernel in detection_kernels:
        # check why no syntax highlighting for convolve2d
        if (convolve2d(board_2d == player_mark, kernel, mode="valid") == 4).any():
            return True
    return False

def score_game(board):
    # game is still ongoing
    reward = None
    game_over = False
    if check_win(board, 1): reward = 1; game_over = True
    # player 1 lost -> guarenteed player 2 win -> by player 2 making winning move
    elif check_win(board, 2): reward = 0; game_over = True
    elif is_tie(board): reward = 0.5; game_over = True
    return (game_over, reward)

def main():
    mcts_utils_test_board = np.array([1, 1, 2, 1, 0, 0, 0,
                                      1, 1, 2, 2, 0, 0, 0,
                                      1, 1, 2, 2, 0, 0, 2,
                                      2, 2, 1, 1, 0, 0, 2,
                                      2, 2, 2, 1, 1, 1, 2,
                                      1, 2, 1, 2, 1, 2, 1])
    # tied board for player 1/2 -> output is 0.5
    mcts_utils_test_board2 = np.array([1, 1, 2, 1, 2, 1, 1,
                                       2, 1, 2, 1, 2, 2, 1,
                                       1, 1, 2, 2, 2, 1, 2,
                                       2, 2, 1, 1, 1, 2, 2,
                                       2, 2, 1, 2, 1, 1, 2,
                                       1, 2, 1, 2, 2, 2, 1])
    # game is not finished yet
    mcts_utils_test_board3 = np.array([0, 0, 0, 0, 0, 0, 0,
                                1, 1, 2, 1, 2, 2, 1,
                                2, 2, 1, 1, 2, 2, 2,
                                1, 1, 2, 2, 2, 1, 1,
                                2, 2, 1, 1, 1, 2, 1,
                                1, 1, 1, 2, 1, 2, 1])
    game_reward = score_game(mcts_utils_test_board3)
    print(f"Game reward from test board:\n{game_reward}")
    

if __name__ == "__main__":
    main()