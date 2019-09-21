from agents import m_learning
import tetris
from tetris.utils import plot_learning_curve
import numpy as np
import os
import time
from datetime import datetime
import random
import multiprocessing
from run import learn_and_evaluate
from run import utils_run

"""

Example run of M-learning with STEW regularization.


"""
parameters = dict(
    # Run parameters
    num_agents=1,  # number of agents that get trained
    test_points=(3, 5, 10, 20, 50, 100, 200, 300),  # iterations of the algorithm in which the agent(s) get(s) tested
    num_games_per_test=10,  # number of test games per 'test_point'. Results are averaged.
    seed=201, # random seed

    # Algorithm parameters
    regularization="stew",
    rollout_length=10,
    number_of_rollouts_per_child=7,
    gamma=0.995,

    learn_from_step=3,
    learn_every_step_until=10,
    learn_periodicity=-7,  # increases by one with every learning step.
    delete_oldest_data_point_every=2,  # the algorithm used the last
    max_batch_size=200,

    dominance_filter=True,
    cumu_dom_filter=True,
    rollout_dom_filter=True,
    rollout_cumu_dom_filter=True,

    lambda_min=-7.0,  # lambda is the regularization parameter
    lambda_max=6,
    num_lambdas=100,

    # Tetris parameters
    num_columns=10,
    num_rows=10,
    feature_type='bcts',
    standardize_features=False,
    max_cleared_test_lines=1000000,

    # Misc parameters
    verbose=False,
    verbose_stew=True)


# INIT
time_id = datetime.now().strftime('%Y_%m_%d_%H_%M'); name_id = "_stew"
run_id = time_id + name_id
run_id_path, models_path, results_path, plots_path = utils_run.create_directories(run_id)
p = utils_run.process_parameters(parameters, run_id_path)
ncpus = np.minimum(multiprocessing.cpu_count(), 2)


def run_loop(p, seed):
    """
    This function contains a complete learning and evaluation run for ONE agent.
    This function is passed multiprocessing.Pool.apply_async() and thus run multiple times (in parallel).

    :param p: `Bunch` of algorithm parameters.
    :param seed: integer, this seed is agent-specific. p also contains a run-specific random seed.
    :return:
    """
    random.seed(seed + p.seed)
    np.random.seed(seed + p.seed)
    agent = m_learning.MLearning(regularization=p.regularization,
                                 dom_filter=p.dominance_filter,
                                 cumu_dom_filter=p.cumu_dom_filter,
                                 rollout_dom_filter=p.rollout_dom_filter,
                                 rollout_cumu_dom_filter=p.rollout_cumu_dom_filter,
                                 lambda_min=p.lambda_min,
                                 lambda_max=p.lambda_max,
                                 num_lambdas=p.num_lambdas,
                                 gamma=p.gamma,
                                 rollout_length=p.rollout_length,
                                 number_of_rollouts_per_child=p.number_of_rollouts_per_child,
                                 learn_every_step_until=p.learn_every_step_until,
                                 learn_from_step_in_current_phase=p.learn_from_step,
                                 max_batch_size=p.max_batch_size,
                                 learn_periodicity=p.learn_periodicity,
                                 num_columns=p.num_columns)
    env = tetris.Tetris(num_columns=p.num_columns, num_rows=p.num_rows)
    test_env = tetris.Tetris(num_columns=p.num_columns, num_rows=p.num_rows, max_cleared_test_lines=p.max_cleared_test_lines)
    test_results_ix, tested_weights_ix = \
        learn_and_evaluate.learn_and_evaluate(env, test_env, agent, p.num_tests,
                                              p.num_games_per_test, p.test_points)
    return [test_results_ix, tested_weights_ix]


time_total_begin = time.time()

# # Execute line below if num_agents == 1
# results = [run_loop(p, 2)]

# Run in parallel (only useful if num_agents > 1)
pool = multiprocessing.Pool(np.minimum(ncpus, p.num_agents))
results = [pool.apply_async(run_loop, (p, seed)) for seed in np.arange(p.num_agents)]

# PROCESS AND SAVE RESULTS
test_results = [results[ix].get()[0] for ix in np.arange(p.num_agents)]
test_results = np.stack(test_results, axis=0)
print("Total time passed: " + str(time.time()-time_total_begin) + " seconds.")
np.save(file=os.path.join(results_path, "test_results.npy"), arr=test_results)


# PLOT LEARNING CURVE
plot_learning_curve(plots_path, test_results, x_axis=p.test_points)

print("Results can be found in directory: " + run_id)

with open(os.path.join(run_id_path, "Info.txt"), "w") as text_file:
    print("USED NUMBER OF CPUS: " + str(ncpus), file=text_file)
    print("Started at: " + time_id + " from file " + str(__file__), file=text_file)
