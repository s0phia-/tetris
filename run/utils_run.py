import os
import json
from tetris.utils import Bunch

def create_directories(run_id):
    run_id_path = os.path.join("experiments", run_id)
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


