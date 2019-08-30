import tetris
from agents.constant_agent import ConstantAgent
import numpy as np
import time
from tetris.utils import print_board_to_string, print_tetromino
from datetime import datetime
from numba import njit

time_id = datetime.now().strftime('%Y_%m_%d_%H_%M')
np.random.seed(1)


@njit
def evaluate(env, agent, num_runs):
    np.random.seed(1)
    rewards = np.zeros(num_runs, dtype=np.int64)
    for i in range(num_runs):
        env.reset()
        while not env.game_over and env.cleared_lines <= env.max_cleared_test_lines:
            print(print_board_to_string(env.current_state))
            print(print_tetromino(env.tetromino_handler.current_tetromino))
            after_state = agent.choose_action_test(start_state=env.current_state, start_tetromino=env.tetromino_handler)
            if after_state.terminal_state:
                print("Game over")
            else:
                pass
                # print(print_tetromino(env.tetromino_handler.current_tetromino))
                # print(print_board_to_string(after_state))
            env.make_step(after_state)

            # if visualize and not env.current_state.terminal_state:
            #     env.print_board_to_string(env.current_state, clear_the_output, sleep)
        print(env.cleared_lines)
        rewards[i] = env.cleared_lines

    return rewards


num_runs = 50
start = time.time()
env = tetris.Tetris(num_columns=10, num_rows=10)

print("Equal weights policy")
agent = ConstantAgent(policy_weights=np.ones(8, dtype=np.float64))
ew_rewards = evaluate(env, agent, num_runs)


print("RANDOM policy")
agent = ConstantAgent(policy_weights=np.random.normal(0, 1, 8))
random_rewards = evaluate(env, agent, num_runs, visualize=False, clear_the_output=False, sleep=0)


print("Canonical non-compensatory weighting (i.e., 1/2, 1/4, 1/8, 1/16, 1/32, ...)")
agent = ConstantAgent(policy_weights=0.5**np.arange(8))
ttb_rewards = evaluate(env, agent, num_runs, visualize=False, clear_the_output=False, sleep=0)

end = time.time()
print("All together took ", end - start, " seconds.")

with open("t" + time_id + ".txt", "w") as text_file:
    print("Started at: " + time_id, file=text_file)  # + " from file " + str(__file__)
    print("Time spent: " + str((end - start)) + "seconds.", file=text_file)
    print("Random Rewards are: " + str(random_rewards), file=text_file)
    print("EW rewards are: " + str(ew_rewards), file=text_file)
    print("EW rewards mean is: " + str(np.mean(ew_rewards)), file=text_file)
    print("TTB rewards are: " + str(ttb_rewards), file=text_file)
    print("TTB rewards mean is: " + str(np.mean(ttb_rewards)), file=text_file)

