from agents import m_learning
import tetris
from tetris.utils import plot_learning_curve
import numpy as np
import os
import time
from datetime import datetime
import random
import multiprocessing
from run import learn
from run import utils_run

time_id = datetime.now().strftime('%Y_%m_%d_%H_%M'); name_id = "_stew"
run_id = time_id + name_id
run_id_path, models_path, results_path, plots_path = utils_run.create_directories(run_id)

param_dict = dict(
    # Run parameters
    num_agents=6,
    test_points=(3, 5, 10, 20, 50, 100, 200, 300),
    # test_points=(3, 5),
    num_test_games=10,
    seed=201,

    # Algorithm parameters
    regularization="stew",
    rollout_length=10,
    avg_expands_per_children=7,
    delete_every=2,
    learn_from_step=3,
    learn_every_step_until=10,
    learn_periodicity=10,
    max_batch_size=200,
    dominance_filter=True,
    cumu_dom_filter=True,
    rollout_dom_filter=True,
    rollout_cumu_dom_filter=True,
    gamma=0.995,
    lambda_min=-8.0,
    lambda_max=4,
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


p = utils_run.process_parameters(param_dict, run_id_path)
ncpus = np.minimum(multiprocessing.cpu_count(), 2)
print("NUMBER OF CPUS: " + str(ncpus))

with open(os.path.join(run_id_path, "cpu_info.txt"), "w") as text_file:
    print("NUMBER OF CPUS: " + str(ncpus), file=text_file)


def run_loop(p, seed):
    random.seed(seed + p.seed)
    np.random.seed(seed + p.seed)
    agent = m_learning.MLearning(regularization=p.regularization,
                                 dominance_filter=p.dominance_filter,
                                 cumu_dom_filter=p.cumu_dom_filter,
                                 rollout_dom_filter=p.rollout_dom_filter,
                                 rollout_cumu_dom_filter=p.rollout_cumu_dom_filter,
                                 lambda_min=p.lambda_min,
                                 lambda_max=p.lambda_max,
                                 num_lambdas=p.num_lambdas,
                                 gamma=p.gamma,
                                 rollout_length=p.rollout_length,
                                 avg_expands_per_children=p.avg_expands_per_children,
                                 learn_every_step_until=p.learn_every_step_until,
                                 learn_from_step=p.learn_from_step,
                                 max_batch_size=p.max_batch_size,
                                 learn_periodicity=p.learn_periodicity,
                                 num_columns=p.num_columns)
    env = tetris.Tetris(num_columns=p.num_columns, num_rows=p.num_rows)
    test_env = tetris.Tetris(num_columns=p.num_columns, num_rows=p.num_rows, max_cleared_test_lines=p.max_cleared_test_lines)
    test_results_ix, tested_weights_ix = \
        learn.learn_and_evaluate(env, test_env, agent, p.num_tests,
                                 p.num_test_games, p.test_points)
    return [test_results_ix, tested_weights_ix]


# run_loop(p, 2)

time_total_begin = time.time()
pool = multiprocessing.Pool(np.minimum(ncpus, p.num_agents))
results = [pool.apply_async(run_loop, (p, seed)) for seed in np.arange(p.num_agents)]

# results = [pool.apply_async(ploops.p_loop, (p, seed, plot_individual)) for seed in np.arange(p.num_agents)]

test_results = [results[ix].get()[0] for ix in np.arange(p.num_agents)]
print("Total time passed: " + str(time.time()-time_total_begin) + " seconds.")


# tested_weights = [results[ix].get()[1] for ix in np.arange(p.num_agents)]
# weights_storage = [results[ix].get()[2] for ix in np.arange(p.num_agents)]
# total_time = [results[ix].get()[3] for ix in np.arange(p.num_agents)]

# test_results = np.mean(np.stack(test_results, axis=0), axis=(0, 2))
test_results = np.stack(test_results, axis=0)


# Save test results
np.save(file=os.path.join(results_path, "test_results.npy"), arr=test_results)

# Save tested_weights
# np.save(file=os.path.join(results_path, "tested_weights.npy"), arr=tested_weights)

# Save tested_weights
# np.save(file=os.path.join(results_path, "weights_storage.npy"), arr=weights_storage)

# Save choice_data
# choice_data = player.mlogit_data.data
# np.save(file=os.path.join(results_path, "choice_data.npy"), arr=choice_data)


###
###  MEAN ANALYSIS
###

plot_learning_curve(plots_path, test_results, x_axis=p.test_points)

# plot_analysis(plots_path, tested_weights, test_results, weights_storage, agent_ix="MEAN")

print("Results can be found in directory: " + run_id)

with open(os.path.join(run_id_path, "Info.txt"), "w") as text_file:
    print("Started at: " + time_id + " from file " + str(__file__), file=text_file)
