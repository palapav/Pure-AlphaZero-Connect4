"""
load most recent pretrained model of alphazero
and pit it against Kaggle negamax agent and if have time -> against our own pure MCTS agent
we will use Kaggle environments
"""
from game_utils import CONNECT4_GRID
from mcts_utils import score_game, play_move, opponent_player_mark, get_illegal_moves
from nn_utils import load_checkpoint
from kaggle_environments import evaluate, make
import torch
import NeuralNetwork
import numpy as np
from MCTS import MCTS

# have self play -> interactions with one self in eval -> no computer generation

"""load latest trained alphazero model here """

def alphazero_agent(observation, configuration):
    """
    takes in observation and outputs move based on p_vector outputted by network
    loaded from memory
    """

    # sampling legal vs illegal actions -> still need to do this
    alphazero_net = NeuralNetwork.AlphaZeroNet()
    # checkpoint number
    alphazero_net = load_checkpoint(alphazero_net, 4)
    board_state = torch.FloatTensor(observation.board)
    policy_estimate = alphazero_net.forward(board_state)
    print("prediction done")
    return torch.argmax(policy_estimate)

def self_eval():
    alphazero_net = NeuralNetwork.AlphaZeroNet()
    alphazero_agent = load_checkpoint(alphazero_net, "a", 2)
    
    connect4_board = np.zeros(CONNECT4_GRID)
    game_over = False

    # initial player mark
    curr_player_mark = 1

    while not game_over:
        """can be refactored"""
        child_priors, value_est = alphazero_agent.forward(MCTS.convert_arr(connect4_board))
        # converting from 1x7 2D tensor to (7,) 1D arr
        child_priors = child_priors.detach().numpy()[0]
        print(f"child priors: {child_priors}")

        # mask illegal moves
        illegal_moves = get_illegal_moves(connect4_board)
        child_priors[illegal_moves] = 0.00000000

        best_move = np.argmax(child_priors)

        play_move(connect4_board, best_move, curr_player_mark)
        print(f"Current Connect 4 board:\n{connect4_board.reshape(6, 7)}")

        game_over, reward = score_game(connect4_board)

        if game_over:
            print(f"Game is over: Player {curr_player_mark} score: {reward}")
            break

        curr_player_mark = opponent_player_mark(curr_player_mark)

        print("Play your move on above board:")
        cli_move = int(input())

        play_move(connect4_board, cli_move, curr_player_mark)

        game_over, reward = score_game(connect4_board)

        if game_over:
            print(f"Game is over: Player {curr_player_mark} score: {1 - reward}")
            break

        curr_player_mark = opponent_player_mark(curr_player_mark)


def eval_single_game():
    # 6 rows and 7 columns in standard connect 4

    # using all of the default parameters
    env = make("connectx", debug='true')

    env.run([alphazero_agent, "negamax"])
    # fix ipython mode
    env.render(mode="ipython")

# separately keep track of value loss
# and policy loss
# Adam optimizer

# look into the board states
# diagonal

def evaluate_agent():
    env = make("connectx", debug='true')
    # print(f"env config: {env.configuration}")
    environment = "connectx"
    steps = []
    agents = [alphazero_agent, "random"]
    num_episodes = 20
    rewards = evaluate(environment, agents, env.configuration, steps, num_episodes)

    print(f"rewards:\n{rewards}")

    num_games = len(rewards)
    win_counter = 0
    draw_counter = 0
    loss_counter = 0

    for i in range(num_games):
        if rewards[i] == [1, -1]:
            win_counter = win_counter + 1
        # opponent's victory (not good)
        elif rewards[i] == [-1, 1]:
            loss_counter = loss_counter + 1
        else:
            draw_counter = draw_counter + 1
    print(f"number of games played: {num_games}")
    print(f"Accuracy of my mcts_agent: {(num_games - loss_counter) / num_games}")
    print("Wins, draws, losses: ", win_counter, draw_counter, loss_counter)


def main():
    # 10 iterations
    #evaluate_agent()
    # eval_single_game()
    self_eval()

if __name__ == '__main__':
    main()