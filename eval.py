"""
load most recent pretrained model of alphazero
and pit it against Kaggle negamax agent and if have time -> against our own pure MCTS agent
we will use Kaggle environments
"""
from nn_utils import load_checkpoint
from kaggle_environments import evaluate, make
import torch
import NeuralNetwork

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

def eval_single_game():
    # 6 rows and 7 columns in standard connect 4

    # using all of the default parameters
    env = make("connectx", debug='true')

    env.run([alphazero_agent, "negamax"])
    env.render(mode="ipython")
    


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
    eval_single_game()

if __name__ == '__main__':
    main()