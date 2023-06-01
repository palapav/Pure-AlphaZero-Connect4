"""
load most recent pretrained model of alphazero
and pit it against Kaggle negamax agent and if have time -> against our own pure MCTS agent
we will use Kaggle environments
"""
from kaggle_environments import evaluate
import torch

"""load latest trained alphazero model here """

def alphazero_agent():
    """
    takes in observation and outputs move based on p_vector outputted by network
    loaded from memory
    """
    pass

def evaluate_agent():
    environment = "connectx"
    steps = []
    agents = [alphazero_agent, "negamax"]
    num_episodes = 10
    rewards = evaluate(environment, agents, steps, num_episodes)

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

    print(f"Accuracy of my mcts_agent: {(num_games - loss_counter) / num_games}")
    print("Wins, draws, losses: ", win_counter, draw_counter, loss_counter)


def main():
    # 10 iterations
    evaluate_agent()

if __name__ == '__main__':
    main()