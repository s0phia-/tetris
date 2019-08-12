import numpy as np
import time
from numba import njit
from agents.constant_agent import ConstantAgent


@njit(cache=False)
def evaluate(env, agent, num_runs):
    # np.random.seed(1)
    rewards = np.zeros(num_runs, dtype=np.int64)
    for i in range(num_runs):
        env.reset()
        while not env.game_over and env.cleared_lines <= env.max_cleared_test_lines:
            after_state = agent.choose_action_test(start_state=env.current_state, start_tetromino=env.tetromino_handler)
            env.make_step(after_state)
            # if visualize and not env.current_state.terminal_state:
            #     env.print_board_to_string(env.current_state, clear_the_output, sleep)
        rewards[i] = env.cleared_lines
    return rewards


def learn_and_evaluate(env,
                       test_env,
                       agent,
                       num_tests,
                       num_test_games,
                       test_points,
                       agent_id=0):
        env.reset()
        test_agent = ConstantAgent(policy_weights=np.ones(env.num_features))
        test_results = np.zeros((num_tests, num_test_games))
        tested_weights = np.zeros((num_tests, env.num_features))
        # weights_storage = np.expand_dims(self.agent.policy_weights, axis=0)
        if num_tests == 0:
            test_index = -1
        else:
            test_index = 0
        while test_index < num_tests:
            # TEST
            if num_tests > 0 and env.cumulative_steps in test_points:
                test_weights = agent.policy_weights.copy()
                test_agent.policy_weights = test_weights
                tested_weights[test_index] = test_weights
                print("tested_weights", tested_weights)
                testing_time_start = time.time()
                print("Agent", agent_id, "is TESTING: ", test_index + 1, " out of ", num_tests, " tests.")
                test_results[test_index, :] = evaluate(test_env, test_agent, num_test_games)
                print("Agent", agent_id, "Mean: ", np.mean(test_results[test_index, :]), ", Median: ", np.median(test_results[test_index, :]))
                # print("Testing took: " + str(time.time() - testing_time_start) + " seconds.")
                test_index += 1
                # testing_time += time.time() - testing_time_start

            # BUILD dataset to LEARN
            #   during learning, if game is over, just restart...
            #   agent can die during learning.
            if env.game_over:
                print("Game was over. Env had to be reset!")
                env.reset()

            if agent.name in ["mlearning"]:
                choosing_action_time_start = time.time()
                after_state, action_index, action_features = agent.choose_action(start_state=env.current_state,
                                                                                 start_tetromino=env.tetromino_handler)
                print("Choosing an action took: " + str(time.time() - choosing_action_time_start) + " seconds.")
                # print("CURRENT STEP: " + str(env.cumulative_steps))
                env.make_step(after_state)
            elif agent.name in ["cbmpi"]:
                env.cumulative_steps += 1

            # LEARN
            if agent.is_learning and not env.game_over:
                # print("Started learning")
                # learning_time_start = time.time()
                if agent.name == "mlearning":
                    agent.learn(action_features=action_features, action_index=action_index)
                    agent.step += 1
                elif agent.name == "cbmpi":
                    agent.learn()
                # print("self.agent.mlogit_data.choice_set_counter: " + str(agent.mlogit_data.choice_set_counter))
                # print("self.agent.mlogit_data.current_number_of_choice_sets: " + str(agent.mlogit_data.current_number_of_choice_sets))

        return test_results, tested_weights # , weights_storage