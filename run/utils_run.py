import os
import json
from tetris.utils import Bunch
from tetris import state
import numpy as np
from numba import njit
import glob


def create_directories(run_id):
    run_id_path = os.path.join("output", run_id)
    print(f"This is the run_id_path {run_id_path}.")
    if not os.path.exists(run_id_path):
        os.makedirs(run_id_path)
    # model_save_name = os.path.join(dir_path, "model.pt")

    models_path = os.path.join(run_id_path, "models")
    if not os.path.exists(models_path):
        os.makedirs(models_path)

    results_path = os.path.join(run_id_path, "results")
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    plots_path = os.path.join(run_id_path, "plots")
    if not os.path.exists(plots_path):
        os.makedirs(plots_path)
    return run_id_path, models_path, results_path, plots_path


def process_parameters(param_dict, run_id_path):
    param_dict["num_tests"] = len(param_dict["test_points"])
    param_dict["plots_path"] = os.path.join(run_id_path, "plots")

    with open(os.path.join(run_id_path, "param_dict.json"), "w") as write_file:
        json.dump(param_dict, write_file, indent=4)

    with open(os.path.join(run_id_path, "param_dict.json"), "r") as read_file:
        param_dict = json.load(read_file)

    p = Bunch(param_dict)
    return p


def load_rollout_state_population(p, max_samples, print_average_height=False):
    sample_list_save_name = p.rollout_population_path
    with open(sample_list_save_name, "r") as ins:
        rollout_population = []
        count = 0
        for x in ins:
            if count < max_samples:
                # print(count)
                rep = np.vstack((np.array([np.array([int(z) for z in bin(int(y))[3:13]]) for y in x.split()]),
                           np.zeros((4, p.num_columns))))
                rep = rep.astype(np.bool_)
                lowest_free_rows = calc_lowest_free_rows(rep)
                rollout_population.append(state.State(rep,
                                                      lowest_free_rows,
                                                      np.array([0], dtype=np.int64),  # changed_lines=
                                                      np.array([0], dtype=np.int64),  # pieces_per_changed_row=
                                                      0.0,  # landing_height_bonus=
                                                      8,  # num_features=
                                                      "bcts",  # feature_type=
                                                      False  # terminal_state=
                                                      ))
                count += 1
            else:
                break

    print(f"Succesfully loaded {count} rollout starting states!")
    if print_average_height:
        average_lowest_free_rows = np.mean([np.mean(d.lowest_free_rows) for d in rollout_population])
        print("average height in rollout state population", average_lowest_free_rows)
    return rollout_population


def clean_up_cmaesout():
    file_list = glob.glob('output/cmaesout*.dat')
    print("Existing cmaes file list", file_list)
    for file_path in file_list:
        try:
            os.remove(file_path)
        except OSError:
            print("Error while deleting file", file_path)


@njit(fastmath=True, cache=False)
def calc_lowest_free_rows(rep):
    num_rows, n_cols = rep.shape
    lowest_free_rows = np.zeros(n_cols, dtype=np.int64)
    for col_ix in range(n_cols):
        lowest = 0
        for row_ix in range(num_rows - 1, -1, -1):
            if rep[row_ix, col_ix] == 1:
                lowest = row_ix + 1
                break
        lowest_free_rows[col_ix] = lowest
    return lowest_free_rows


