CONNECT4_GRID = 42
EMPTY = 0
COLUMNS = 7
ROWS = 6

#---- methods to facilitate MCTS simulations-------
# will have to refactor later into MCTS utils class most likely
# reduce redundant design between game_utils and mcts_utils
# make design decision of object oriented vs functional design
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
    
# terrible software design right now -> need to refactor tomorrow
# playing the selected move on the game board for an episode per epoch
# we train the network after each epoch
def play_move(board, column, player_mark):
    row = max([r for r in range(ROWS) if board[column + (r * COLUMNS)] == EMPTY])
    board[column + (row * COLUMNS)] = player_mark

# All in terms of player 1
# 0, 0.5, 1 win for player 1
# if it is not a win/tie/loss for player 1 -> game is still continuing
# return None for reward
def is_win(board, column, player_mark):
    inarow = 3
    target_rows = [r for r in range(ROWS) if board[column + (r * COLUMNS)] == player_mark]
    if len(target_rows) == 0: return False
    row =  min(target_rows)
    
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


def is_tie(board):
    return not(any(mark == EMPTY for mark in board[0: COLUMNS + 1]))
    
# we score the game prior to player_mark placing mark on board
def score_game(board, column, player_mark):
    if is_win(board, column, player_mark):
        return (True, 1)
    if is_tie(board):
        return (True, 0)
    else:
        # opponent won -> current player -> -1 score in backprop
        # opponent has won or game not finished yet
        return (False, None)

def main(): pass

if __name__ == "__main__":
    main()