
def evaluate(env, agent, hard_test=False, visualize=False, clear_the_output=False, sleep=0.01):
    env.reset()
    # env.print_board_to_string(env.current_state, clear_the_output, sleep)
    while not env.game_over and env.cleared_lines <= env.max_cleared_test_lines:
        current_tetromino = env.tetromino_sampler.next_tetromino()
        if hard_test:
            chosen_action = agent.choose_action_test_hard(env.current_state, current_tetromino)
        else:
            chosen_action, move_index = agent.choose_action_test(start_state=env.current_state, start_tetromino=current_tetromino)
        env.make_step(chosen_action)
        if visualize and not env.current_state.terminal_state:
            env.print_board_to_string(env.current_state, clear_the_output, sleep)
    return env.cleared_lines


import tetris
from agents.constant_agent import ConstantAgent
# from run.evaluate import evaluate
import numpy as np
import time
from datetime import datetime
time_id = datetime.now().strftime('%Y_%m_%d_%H_%M')
np.random.seed(1)
start = time.time()
# rewards = np.zeros(2)
# for i in range(2):
rewards = np.zeros(10)
for i in range(10):
    env = tetris.Tetris(num_columns=10, num_rows=10, verbose=True)
    agent = ConstantAgent(policy_weights=np.ones(8))
    rewards[i] = evaluate(env, agent, visualize=False, clear_the_output=False, sleep=0)

end = time.time()
print("Took ", end - start, " seconds.")


with open("t" + time_id + ".txt", "w") as text_file:
    print("Started at: " + time_id, file=text_file)  # + " from file " + str(__file__)
    print("Time spent: " + str((end-start)) + "seconds.", file=text_file)
    print("Rewards are: " + str(rewards), file=text_file)

# import tetris
# from agents.m_learning import MLearning
# from agents.random_agent import RandomAgent
# from run.evaluate import evaluate
# env = tetris.Tetris(num_columns=10, num_rows=10, verbose=True)
# agent = RandomAgent()
#
# evaluate(env, agent, visualize=True, clear_the_output=True)


