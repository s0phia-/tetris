import tetris
from agents.constant_agent import ConstantAgent
# from run.evaluate import evaluate
import numpy as np
import time
from datetime import datetime

time_id = datetime.now().strftime('%Y_%m_%d_%H_%M')
np.random.seed(1)

import numba.config as c
print("NUMBA_DISABLE_JIT:", c.DISABLE_JIT)
c.DISABLE_JIT = 0
# print("NUMBA_DISABLE_JIT:", c.DISABLE_JIT)


def evaluate(env, agent, visualize=False, clear_the_output=False, sleep=0):
    env.reset()
    # _ = env.print_board_to_string(env.current_state, clear_the_output, sleep)
    while not env.game_over and env.cleared_lines <= env.max_cleared_test_lines:
        current_tetromino = env.tetromino_sampler.next_tetromino()
        # print(current_tetromino)
        chosen_action, move_index = agent.choose_action_test(start_state=env.current_state, start_tetromino=current_tetromino)
        env.make_step(chosen_action)
        # assert not np.any(env.current_state.representation[] > env.num_rows)
        if visualize and not env.current_state.terminal_state:
            env.print_board_to_string(env.current_state, clear_the_output, sleep)
    return env.cleared_lines

start = time.time()

# print("RANDOM")
# random_rewards = np.zeros(10)
# for i in range(10):
#     env = tetris.Tetris(num_columns=10, num_rows=10, verbose=True)
#     agent = ConstantAgent(policy_weights=np.random.normal(0, 1, 8))
#     random_rewards[i] = evaluate(env, agent, visualize=False, clear_the_output=False, sleep=0)

print("EW")
rewards = np.zeros(100)
for i in range(100):
    env = tetris.Tetris(num_columns=10, num_rows=10, verbose=True)
    agent = ConstantAgent(policy_weights=np.ones(8, dtype=np.float64))
    rewards[i] = evaluate(env, agent, visualize=False, clear_the_output=False, sleep=0)

end = time.time()
print("Took ", end - start, " seconds.")

with open("t" + time_id + ".txt", "w") as text_file:
    print("Started at: " + time_id, file=text_file)  # + " from file " + str(__file__)
    print("Time spent: " + str((end - start)) + "seconds.", file=text_file)
    # print("Random Rewards are: " + str(random_rewards), file=text_file)
    print("Rewards are: " + str(rewards), file=text_file)
    print("Rewards mean is: " + str(np.mean(rewards)), file=text_file)

