import numpy as np
import matplotlib.pyplot as plt
import os
from numba import njit
# import telegram
from os.path import expanduser
home = expanduser("~")


# def notify_ending(message):
#     print("NOTIFY ENDING is DEPRECATED!")
#     with open(os.path.join(home, "tel_keys.json"), 'r') as keys_file:
#         k = json.load(keys_file)
#         token = k['telegram_token']
#         chat_id = k['telegram_chat_id']
#
#     bot = telegram.Bot(token=token)
#     bot.sendMessage(chat_id=chat_id, text=message)


class Bunch(object):
    def __init__(self, adict):
        self.__dict__.update(adict)


def print_board_to_string(state):
    string = "\n"
    for row_ix in range(state.representation.shape[0]):
        # Start from top
        row_ix = state.num_rows - row_ix - 1
        string += "|"
        for col_ix in range(state.num_columns):
            if state.representation[row_ix, col_ix]:
                string += "██"
            else:
                string += "  "
        string += "|\n"
    return string


def print_tetromino(tetromino_index):
    if tetromino_index == 0:
        string = '''
██ ██ ██ ██'''
    elif tetromino_index == 1:
        string = '''
██ ██ 
██ ██'''
    elif tetromino_index == 2:
        string = '''
   ██ ██ 
██ ██'''
    elif tetromino_index == 3:
        string ='''
██ ██ 
   ██ ██'''
    elif tetromino_index == 4:
        string ='''
   ██
██ ██ ██'''
    elif tetromino_index == 5:
        string ='''
██ ██ ██
██'''
    elif tetromino_index == 6:
        string="""
██ ██ ██
      ██"""
    return string

@njit(cache=False)
def one_hot_vector(one_index, length):
    out = np.zeros(length)
    out[one_index] = 1.
    return out


@njit(cache=False)
def vert_one_hot(one_index, length):
    out = np.zeros((length, 1))
    out[one_index] = 1.
    return out


@njit(cache=False)
def compute_action_probabilities(action_features, weights, temperature):
    utilities = action_features.dot(weights) / temperature
    utilities = utilities - np.max(utilities)
    exp_utilities = np.exp(utilities)
    probabilities = exp_utilities / np.sum(exp_utilities)
    return probabilities


@njit(cache=False)
def grad_of_log_action_probabilities(features, probabilities, action_index):
    features_of_chosen_action = features[action_index]
    grad = features_of_chosen_action - features.T.dot(probabilities)
    return grad


@njit(cache=False)
def softmax(U):
    ps = np.exp(U - np.max(U))
    ps /= np.sum(ps)
    return ps


def plot_multiple_learning_curves(plots_path, compare_results, compare_ids, x_axis):
    # Plot and save learning curves.
    fig1, ax1 = plt.subplots()
    for test_results_ix in range(len(compare_results)):
        test_results = compare_results[test_results_ix]
        mean_array = np.mean(test_results, axis=(0, 2))
        serr_array = np.std(test_results, axis=(0, 2)) / np.sqrt(compare_results[0].shape[0])
        ax1.plot(x_axis, mean_array, label=compare_ids[test_results_ix])
        ax1.fill_between(x_axis, mean_array - serr_array, mean_array + serr_array, alpha=0.2)

    plt.title('Mean performance')
    plt.xlabel('Iteration')
    plt.ylabel('Mean score')
    plt.legend()
    plt.show()
    fig1.savefig(os.path.join(plots_path, "mean_performance"))
    plt.close()

    fig1, ax1 = plt.subplots()
    for test_results_ix in range(len(compare_results)):
        test_results = compare_results[test_results_ix]
        mean_array = np.median(test_results, axis=(0, 2))
        ax1.plot(x_axis, mean_array, label=compare_ids[test_results_ix])

    plt.title('Median performance')
    plt.xlabel('Iteration')
    plt.ylabel('Median score')
    plt.legend()
    plt.show()
    fig1.savefig(os.path.join(plots_path, "median_performance"))
    plt.close()

    fig1, ax1 = plt.subplots()
    for test_results_ix in range(len(compare_results)):
        test_results = compare_results[test_results_ix]
        mean_array = np.max(test_results, axis=(0, 2))
        ax1.plot(x_axis, mean_array, label=compare_ids[test_results_ix])

    plt.title('Max performance')
    plt.xlabel('Iteration')
    plt.ylabel('Max score')
    plt.legend()
    plt.show()
    fig1.savefig(os.path.join(plots_path, "max_performance"))
    plt.close()


def plot_learning_curve(plots_path, test_results, x_axis, suffix=""):

    mean_array = np.mean(test_results, axis=(0, 2))
    median_array = np.median(test_results, axis=(0, 2))
    max_array = np.max(test_results, axis=(0, 2))
    serr_array = np.std(test_results, axis=(0, 2)) / np.sqrt(test_results[0].shape[0])


    # Plot and save learning curves.
    fig1, ax1 = plt.subplots()
    ax1.plot(x_axis, mean_array, label="mean")
    # ax1.plot(x_axis, median_array, label="median")
    ax1.fill_between(x_axis, mean_array - serr_array, mean_array + serr_array, alpha=0.2)
    plt.legend()
    fig1.show()
    fig1.savefig(os.path.join(plots_path, "mean_performance" + suffix))
    plt.close()

    # Plot and save learning curves.
    fig1, ax1 = plt.subplots()
    ax1.plot(x_axis, max_array, label="max")
    plt.legend()
    fig1.show()
    fig1.savefig(os.path.join(plots_path, "max_performance"+ suffix))
    plt.close()


def plot_individual_agent(plots_path, tested_weights, test_results, agent_ix, x_axis):
    feature_names = ['rows_with_holes', 'column_transitions', 'holes', 'landing_height', 'cumulative_wells',
                     'row_transitions', 'eroded', 'hole_depth']
    # Compute tested_weight paths
    if tested_weights is not None:
        fig1, ax1 = plt.subplots()
        for ix in range(tested_weights.shape[1]):
            ax1.plot(x_axis, tested_weights[:, ix], label=feature_names[ix])
        plt.legend()
        fig1.show()
        fig1.savefig(os.path.join(plots_path, "weight_paths_tested" + str(agent_ix)))
        plt.close()

    # # Compute weights_storage paths
    # fig1, ax1 = plt.subplots()
    # for ix in range(weights_storage.shape[1]):
    #     ax1.plot(weights_storage[:, ix], label=feature_names[ix])
    # plt.legend()
    # fig1.show()
    # fig1.savefig(os.path.join(plots_path, "weight_paths" + str(agent_ix)))
    # plt.close()

    # # Compute weight distances
    # # tested_weights = np.random.normal(size=(4, 8))
    # diff_weights = np.diff(tested_weights, axis=0)
    # distances = np.sqrt(np.sum(diff_weights ** 2, axis=1))
    # fig1, ax1 = plt.subplots()
    # ax1.plot(distances, label="l2 distance to previous")
    # plt.legend()
    # fig1.show()
    # fig1.savefig(os.path.join(plots_path, "distances" + str(agent_ix)))
    # plt.close()

    # # Compute RELATIVE weight distances
    # relative_diff_weights = np.diff(tested_weights / np.abs(tested_weights[:, 0][:, np.newaxis]), axis=0)
    # distances = np.sqrt(np.sum(relative_diff_weights ** 2, axis=1))
    # fig1, ax1 = plt.subplots()
    # ax1.plot(distances, label="l2 RELATIVE distance to previous")
    # plt.legend()
    # fig1.show()
    # fig1.savefig(os.path.join(plots_path, "relative_distances" + str(agent_ix)))
    # plt.close()

    # Compute learning curves (mean and median)
    mean_array = np.mean(test_results, axis=1)
    median_array = np.median(test_results, axis=1)
    # max_array = np.max(test_results, axis=1)

    # Plot and save learning curves.
    fig1, ax1 = plt.subplots()
    ax1.plot(x_axis, mean_array, label="mean")
    ax1.plot(x_axis, median_array, label="median")
    plt.legend()
    fig1.show()
    fig1.savefig(os.path.join(plots_path, "mean_performance" + str(agent_ix)))
    plt.close()

    # # Plot and save learning curves.
    # fig1, ax1 = plt.subplots()
    # ax1.plot(max_array, label="max")
    # plt.legend()
    # fig1.show()
    # fig1.savefig(os.path.join(plots_path, "max_performance" + str(agent_ix)))
    # plt.close()


def plot_analysis(plots_path, tested_weights, test_results, weights_storage, agent_ix, x_axis):
    feature_names = ['rows_with_holes', 'column_transitions', 'holes', 'landing_height', 'cumulative_wells',
                     'row_transitions', 'eroded', 'hole_depth']
    # Compute tested_weight paths
    fig1, ax1 = plt.subplots()
    for ix in range(tested_weights.shape[1]):
        ax1.plot(x_axis, tested_weights[:, ix], label=feature_names[ix])
    plt.legend()
    fig1.show()
    fig1.savefig(os.path.join(plots_path, "weight_paths_tested" + str(agent_ix)))
    plt.close()

    # Compute weights_storage paths
    fig1, ax1 = plt.subplots()
    for ix in range(weights_storage.shape[1]):
        ax1.plot(weights_storage[:, ix], label=feature_names[ix])
    plt.legend()
    fig1.show()
    fig1.savefig(os.path.join(plots_path, "weight_paths" + str(agent_ix)))
    plt.close()

    # Compute weight distances
    # tested_weights = np.random.normal(size=(4, 8))
    diff_weights = np.diff(tested_weights, axis=0)
    distances = np.sqrt(np.sum(diff_weights ** 2, axis=1))
    fig1, ax1 = plt.subplots()
    ax1.plot(distances, label="l2 distance to previous")
    plt.legend()
    fig1.show()
    fig1.savefig(os.path.join(plots_path, "distances" + str(agent_ix)))
    plt.close()

    # Compute RELATIVE weight distances
    relative_diff_weights = np.diff(tested_weights / np.abs(tested_weights[:, 0][:, np.newaxis]), axis=0)
    distances = np.sqrt(np.sum(relative_diff_weights ** 2, axis=1))
    fig1, ax1 = plt.subplots()
    ax1.plot(distances, label="l2 RELATIVE distance to previous")
    plt.legend()
    fig1.show()
    fig1.savefig(os.path.join(plots_path, "relative_distances" + str(agent_ix)))
    plt.close()

    # Compute learning curves (mean and median)
    mean_array = np.mean(test_results, axis=1)
    median_array = np.median(test_results, axis=1)
    max_array = np.max(test_results, axis=1)

    # Plot and save learning curves.
    fig1, ax1 = plt.subplots()
    ax1.plot(x_axis, mean_array, label="mean")
    ax1.plot(x_axis, median_array, label="median")
    plt.legend()
    fig1.show()
    fig1.savefig(os.path.join(plots_path, "mean_performance" + str(agent_ix)))
    plt.close()

    # Plot and save learning curves.
    fig1, ax1 = plt.subplots()
    ax1.plot(x_axis, max_array, label="max")
    plt.legend()
    fig1.show()
    fig1.savefig(os.path.join(plots_path, "max_performance" + str(agent_ix)))
    plt.close()


