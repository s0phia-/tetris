import numpy as np
import time
from numba import njit
from agents.constant_agent import ConstantAgent
from tetris.utils import print_board_to_string, print_tetromino


def learn_and_evaluate(env,
                       test_env,
                       agent,
                       num_tests,
                       num_test_games,
                       test_points,
                       agent_id=0,
                       store_weights=False):
        env.reset()
        if agent.name == "cbmpi":
            test_agent = ConstantAgent(policy_weights=np.ones(env.num_features, dtype=np.float64),
                                       feature_directors=np.ones(env.num_features, dtype=np.float64))
        else:
            test_agent = ConstantAgent(policy_weights=np.ones(env.num_features, dtype=np.float64),
                                       feature_directors=2*(np.random.binomial(1, 0.5, 8) - 0.5))
        test_results = np.zeros((num_tests, num_test_games))
        tested_weights = np.zeros((num_tests, env.num_features))
        if store_weights:
            weights_storage = np.expand_dims(agent.policy_weights * agent.copy_current_feature_directors(), axis=0)
        if num_tests == 0:
            test_index = -1
        else:
            test_index = 0
        while test_index < num_tests:
            print("agent.step: ", agent.step)
            # print("agent.step_in_current_phase: ", agent.step_in_current_phase)
            # TEST
            if num_tests > 0 and agent.step in test_points:
                test_weights = agent.policy_weights.copy()
                test_agent.policy_weights = test_weights

                if agent.name == "hierarchical_learning":
                    if agent.num_phases == 1 and agent.current_phase == "learning_weights" and agent.step_in_current_phase <= agent.learn_from_step_in_current_phase:
                        test_directors = np.zeros(env.num_features, dtype=np.float64)
                    else:
                        test_directors = agent.copy_current_feature_directors()
                    test_agent.feature_directors = test_directors
                    print("test_directors", test_directors)
                    print("test_weights * test_directors", test_weights * test_directors)
                tested_weights[test_index] = test_weights
                # print("tested_weights", tested_weights)

                # testing_time_start = time.time()
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

            if agent.name in ["mlearning", "hierarchical_learning"]:
                # print("test_directors", agent.copy_current_feature_directors())
                # print("test_weights * test_directors", agent.policy_weights.copy() * agent.copy_current_feature_directors())
                # print(print_board_to_string(env.current_state))
                # print(print_tetromino(env.tetromino_handler.current_tetromino))
                choosing_action_time_start = time.time()
                after_state, action_index, action_features = agent.choose_action(start_state=env.current_state,
                                                                                 start_tetromino=env.tetromino_handler)
                # print("Choosing an action took: " + str(time.time() - choosing_action_time_start) + " seconds.")
                # print("CURRENT STEP: " + str(env.cumulative_steps))
                env.make_step(after_state)
            elif agent.name in ["cbmpi"]:
                env.cumulative_steps += 1
            else:
                raise ValueError("'agent.name' has to be either 'mlearning', 'hierarchical_learning', or 'cbmpi'")

            # LEARN
            if agent.is_learning and not env.game_over:
                # print("Started learning")
                # learning_time_start = time.time()
                if agent.name in ["mlearning", "hierarchical_learning"]:
                    agent.append_data(action_features=action_features, action_index=action_index)
                    agent.learn(action_features=action_features, action_index=action_index)
                    if store_weights:
                        weights_storage = np.vstack((weights_storage, agent.policy_weights.copy() * agent.copy_current_feature_directors()))
                elif agent.name == "cbmpi":
                    agent.learn()
                # print("self.agent.mlogit_data.choice_set_counter: " + str(agent.mlogit_data.choice_set_counter))
                # print("self.agent.mlogit_data.current_number_of_choice_sets: " + str(agent.mlogit_data.current_number_of_choice_sets))
            agent.update_steps()
        if store_weights:
            return test_results, tested_weights, weights_storage
        else:
            return test_results, tested_weights  # , weights_storage


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
