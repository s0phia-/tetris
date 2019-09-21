import numpy as np
from numba import njit
from agents.constant_agent import ConstantAgent


def learn_and_evaluate(env,
                       test_env,
                       agent,
                       test_agent,
                       test_points,
                       num_games_per_test,
                       agent_id=0):
    num_tests = len(test_points)
    env.reset()
    test_results = np.zeros((num_tests, num_games_per_test))
    test_index = 0
    while test_index < num_tests:
        print("agent.step: ", agent.step)

        # TEST
        if num_tests > 0 and agent.step in test_points:
            test_agent.policy_weights = agent.copy_current_policy_weights()
            test_results[test_index, :] = evaluate(test_env, test_agent, num_games_per_test)
            print("Agent", agent_id, "was testing: ", test_index + 1, " out of ", num_tests, " tests. \n Mean: ", np.mean(test_results[test_index, :]), ", Median: ", np.median(test_results[test_index, :]))
            test_index += 1

        # LEARN
        # Make step in the environment (NOT needed for CBMPI / BatchRollout)
        if agent.rollout_handler.name == "OnlineRollout":
            if env.game_over:  # This is only relevant for online algorithms (does not have a side-effect on CBMPI-type algos, though.)
                print("Game was over. Env had to be reset!")
                env.reset()
            after_state, action_index, action_features = agent.choose_action(start_state=env.current_state,
                                                                             start_tetromino=env.generative_model)
            env.make_step(after_state)
            agent.learn(action_features=action_features, action_index=action_index)
        elif agent.rollout_handler.name == "BatchRollout":
            agent.learn()

        agent.update_steps()
    return test_results


def learn_and_evaluate_old(env,
                           test_env,
                           agent,
                           num_tests,
                           num_games_per_test,
                           test_points,
                           agent_id=0,
                           store_weights=False):

        """
        Older, more involved version that allows to keep track of tested weights.
        """
        assert(agent.name in ["mlearning", "hierarchical_learning", "cbmpi"])
        env.reset()
        if agent.name == "cbmpi":
            test_agent = ConstantAgent(policy_weights=np.ones(env.num_features, dtype=np.float64),
                                       feature_directors=np.ones(env.num_features, dtype=np.float64))
        else:
            test_agent = ConstantAgent(policy_weights=np.ones(env.num_features, dtype=np.float64),
                                       feature_directors=2*(np.random.binomial(1, 0.5, 8) - 0.5))
        test_results = np.zeros((num_tests, num_games_per_test))
        tested_weights = np.zeros((num_tests, env.num_features))
        if store_weights:
            weights_storage = np.expand_dims(agent.policy_weights * agent.copy_current_feature_directors(), axis=0)
        if num_tests == 0:
            test_index = -1
        else:
            test_index = 0
        while test_index < num_tests:
            print("agent.step: ", agent.step)
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
                print("Agent", agent_id, "is TESTING: ", test_index + 1, " out of ", num_tests, " tests.")
                test_results[test_index, :] = evaluate(test_env, test_agent, num_games_per_test)
                print("Agent", agent_id, "Mean: ", np.mean(test_results[test_index, :]), ", Median: ", np.median(test_results[test_index, :]))
                test_index += 1

            # Make step in the environment (not needed for cbmpi)
            # During learning, if game is over, just restart... agent can die during learning.
            if agent.name in ["mlearning", "hierarchical_learning"]:
                if env.game_over:
                    print("Game was over. Env had to be reset!")
                    env.reset()
                after_state, action_index, action_features = agent.choose_action(start_state=env.current_state,
                                                                                 start_tetromino=env.generative_model)
                env.make_step(after_state)

            # LEARN
            if agent.is_learning and not env.game_over:
                if agent.name in ["mlearning", "hierarchical_learning"]:
                    agent.append_data(action_features=action_features, action_index=action_index)
                    agent.learn(action_features=action_features, action_index=action_index)
                elif agent.name == "cbmpi":
                    agent.learn()
                if store_weights:
                    weights_storage = np.vstack((weights_storage, agent.policy_weights.copy() * agent.copy_current_feature_directors()))
            agent.update_steps()
        if store_weights:
            return test_results, tested_weights, weights_storage
        else:
            return test_results, tested_weights  # , weights_storage


# @njit(cache=False)
def evaluate(env, agent, num_runs):
    rewards = np.zeros(num_runs, dtype=np.int64)
    for i in range(num_runs):
        env.reset()
        while not env.game_over and env.cleared_lines <= env.max_cleared_test_lines:
            after_state = agent.choose_action_test(start_state=env.current_state, start_tetromino=env.generative_model)
            env.make_step(after_state)
        rewards[i] = env.cleared_lines
    return rewards
