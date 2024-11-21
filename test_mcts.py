from kaggle_environments import evaluate, make
import numpy as np


# local imports
from mcts import MCTS
from nn import AlphaZeroNet

def get_reward(outcomes):
    num_games = outcomes.shape[0]
    # record number of wins, losses, and ties
    first_agent_outcomes = outcomes[:, 0]
    num_wins = (first_agent_outcomes == 1).sum()
    num_ties = (first_agent_outcomes == 0).sum()
    num_losses = (first_agent_outcomes == -1).sum()
    return num_wins, num_ties, num_losses

def MCTS_agent(observation, configuration):
    "Follows the Kaggle Environment specs to be used in the evaluate method"
    alphazero_nn = AlphaZeroNet()
    mcts = MCTS()

    root_player_mark = 1
    training_dataset = []
    player_one_move = mcts.search(alphazero_nn, 500, root_player_mark, np.array(observation.board, dtype=np.int32), training_dataset)

    return player_one_move



# Run multiple episodes to estimate its performance.
p1_name = MCTS_agent
p2_name = "negamax"
outcomes = evaluate("connectx", [p1_name, p2_name], num_episodes=10)
p1_wins, p1_ties, p1_losses = get_reward(np.array(outcomes, dtype=np.int8))
print(f"{p1_name}: Wins: {p1_wins}, Ties: {p1_ties}, Losses: {p1_losses}")

# env = make("connectx", debug=True)
# configuration = env.configuration

# env.run([MCTS_agent, p2_name])
# print("Success!" if env.state[0].status == env.state[1].status == "DONE" else "Failed...")