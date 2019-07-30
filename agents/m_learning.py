import numpy as np
from domtools import dom_filter
from tetris import tetromino
from stew import StewMultinomialLogit, ChoiceSetData
from stew.utils import create_diff_matrix, create_ridge_matrix
from tetris.state import State
from numba import njit
import time


class MLearning:
    """
    M-learning, tailored to application in Tetris.
    """
    def __init__(self,
                 regularization,
                 dominance_filter,
                 cumu_dom_filter,
                 rollout_dom_filter,
                 rollout_cumu_dom_filter,
                 lambda_min,
                 lambda_max,
                 num_lambdas,
                 gamma,
                 rollout_length,
                 number_of_rollouts_per_child,
                 learn_every_step_until,
                 max_batch_size,
                 learn_periodicity,
                 learn_from_step,
                 num_columns,
                 feature_directors=np.array([-1, -1, -1, -1, -1, -1, 1, -1]),
                 feature_type="bcts",
                 verbose=False,
                 verbose_stew=False):

        # Tetris params
        self.num_columns = num_columns  # ...of the Tetris board
        self.feature_type = feature_type
        self.num_features = 8
        self.feature_names = np.array(['rows_with_holes', 'column_transitions', 'holes',
                                       'landing_height', 'cumulative_wells', 'row_transitions',
                                       'eroded', 'hole_depth'])  # Uses BCTS features.
        self.verbose = verbose
        self.verbose_stew = verbose_stew
        self.max_choice_set_size = 34  # There are never more than 34 actions in Tetris
        self.tetromino_handler = tetromino.Tetromino(self.feature_type, self.num_features, self.num_columns)

        # Algo params
        self.gamma = gamma
        self.rollout_length = rollout_length
        self.number_of_rollouts_per_child = number_of_rollouts_per_child
        self.num_total_rollouts = self.rollout_length * self.number_of_rollouts_per_child
        self.dom_filter = dominance_filter
        self.cumu_dom_filter = cumu_dom_filter
        self.rollout_dom_filter = rollout_dom_filter
        self.rollout_cumu_dom_filter = rollout_cumu_dom_filter

        # Algo init
        self.policy_weights = np.random.normal(loc=0.0, scale=0.1, size=self.num_features)
        self.feature_directors = feature_directors
        self.feature_order = np.arange(self.num_features)
        self.step = 0

        # Data and model (regularization type)
        self.regularization = regularization
        assert(self.regularization in ["stew", "ols", "ridge", "nonnegative", "ew", "ttb"])
        self.is_learning = self.regularization in ["stew", "ridge", "ols", "nonnegative"]
        if self.regularization == "ridge":
            D = create_ridge_matrix(self.num_features)
        else:
            D = create_diff_matrix(self.num_features)
        self.model = StewMultinomialLogit(num_features=self.num_features, D=D, lambda_min=lambda_min,
                                          lambda_max=lambda_max, num_lambdas=num_lambdas, verbose=self.verbose_stew,
                                          nonnegative=self.regularization=="nonnegative")
        self.mlogit_data = ChoiceSetData(num_features=self.num_features, max_choice_set_size=self.max_choice_set_size)

        # Algo batch size handling
        self.delete_oldest_data_point_every = 2
        self.learn_from_step = learn_from_step
        self.learn_periodicity = learn_periodicity
        self.learn_every_step_until = learn_every_step_until
        self.max_batch_size = max_batch_size
        self.step_since_last = 0

    # def reset_agent(self):
    #     self.step = 0

    # def create_rollout_tetrominos(self):
    #     self.rollout_tetrominos = np.array([self.tetromino_sampler.next_tetromino() for _ in range(self.num_total_rollouts)])
    #     self.rollout_tetrominos.shape = (self.rollout_length, self.number_of_rollouts_per_child)

    def choose_action(self, start_state, start_tetromino):
        return choose_action_using_rollouts(start_state, start_tetromino,
                                            self.rollout_length, self.tetromino_handler, self.policy_weights,
                                            self.dom_filter, self.cumu_dom_filter, self.rollout_dom_filter, self.rollout_cumu_dom_filter,
                                            self.feature_directors, self.num_features, self.gamma,
                                            self.number_of_rollouts_per_child)

    def learn(self, action_features, action_index):
        """
        Learns new policy weights from choice set data.
        """
        delete_oldest = self.mlogit_data.current_number_of_choice_sets > self.max_batch_size or (self.delete_oldest_data_point_every > 0 and self.step % self.delete_oldest_data_point_every == 0 and self.step >= self.learn_from_step + 1)
        self.mlogit_data.push(features=action_features, choice_index=action_index, delete_oldest=delete_oldest)
        self.step_since_last += 1
        if self.step >= self.learn_from_step and (self.step <= self.learn_every_step_until or self.step_since_last == self.learn_periodicity):
            self.learn_periodicity += 1
            print("self.learn_periodicity", self.learn_periodicity)
            print("Started learning")
            learning_time_start = time.time()
            if self.regularization in ["ols", "nonnegative"]:
                self.policy_weights = self.model.fit(data=self.mlogit_data.sample(), lam=0, standardize=False)
            elif self.regularization in ["ridge", "stew"]:
                self.policy_weights, _ = self.model.cv_fit(data=self.mlogit_data.sample())
            print("Learning took: " + str(time.time() - learning_time_start) + " seconds.")
            self.step_since_last = 0


@njit(cache=False)
def choose_action_using_rollouts(start_state, start_tetromino,
                                 rollout_length, tetromino_handler, policy_weights,
                                 dominance_filter, cumu_dom_filter, rollout_dom_filter, rollout_cumu_dom_filter,
                                 feature_directors, num_features, gamma,
                                 number_of_rollouts_per_child):
    children_states = start_tetromino.get_after_states(start_state)
    num_children = len(children_states)
    if num_children == 0:
        # Game over!
        return State(np.zeros((1, 1), dtype=np.bool_), np.zeros(1, dtype=np.int64),
                     np.array([0], dtype=np.int64), np.array([0], dtype=np.int64),
                     0.0, 1, "bcts", True), 0, np.zeros((2, 2))

    action_features = np.zeros((num_children, num_features), dtype=np.float_)
    for ix in range(num_children):
        action_features[ix] = children_states[ix].get_features(feature_directors, False)  # , order_by=self.feature_order
    if dominance_filter or cumu_dom_filter:
        not_simply_dominated, not_cumu_dominated = dom_filter(action_features, len_after_states=num_children)  # domtools.
    #     if cumu_dom_filter:
    #         children_states = children_states[not_cumu_dominated]
    #         map_back_vector = np.nonzero(not_cumu_dominated)[0]
    #     else:  # Only simple dominance
    #         children_states = children_states[not_simply_dominated]
    #         map_back_vector = np.nonzero(not_simply_dominated)[0]
    #     num_children = len(children_states)
    # else:
    #     map_back_vector = np.arange(num_children)
    child_total_values = np.zeros(num_children)
    # self.create_rollout_tetrominos()
    for child in range(num_children):
        # TODO: ONLY WORKS WITH cumu_dom_filter ON
        if not_cumu_dominated[child]:
            for rollout_ix in range(number_of_rollouts_per_child):
                child_total_values[child] += roll_out(children_states[child], rollout_length, tetromino_handler, policy_weights,
                                                      rollout_dom_filter, rollout_cumu_dom_filter,
                                                      feature_directors, num_features, gamma)
        else:
            child_total_values[child] = -np.inf
    child_index = np.argmax(child_total_values)
    # children_states[child_index].value_estimate = child_total_values[child_index]
    # before_filter_index = map_back_vector[child_index]  # Needed for probabilities in gradient in learn()
    return children_states[child_index], child_index, action_features


@njit(cache=False)
def roll_out(start_state, rollout_length, tetromino_handler, policy_weights,
             rollout_dom_filter, rollout_cumu_dom_filter,
             feature_directors, num_features, gamma):
    value_estimate = start_state.n_cleared_lines
    state_tmp = start_state
    count = 1
    while not state_tmp.terminal_state and count <= rollout_length:
        tetromino_handler.next_tetromino()
        available_after_states = tetromino_handler.get_after_states(state_tmp)
        if len(available_after_states) == 0:
            # Game over!
            return value_estimate
        state_tmp = choose_action_in_rollout(available_after_states, policy_weights,
                                             rollout_dom_filter, rollout_cumu_dom_filter,
                                             feature_directors, num_features)
        value_estimate += gamma ** count * state_tmp.n_cleared_lines
        count += 1
    return value_estimate


@njit(cache=False)
def choose_action_in_rollout(available_after_states, policy_weights,
                             rollout_dom_filter, rollout_cumu_dom_filter,
                             feature_directors, num_features):
    num_states = len(available_after_states)
    action_features = np.zeros((num_states, num_features))
    for ix, after_state in enumerate(available_after_states):
        action_features[ix] = after_state.get_features(feature_directors, False)  # , order_by=self.feature_order
    if rollout_cumu_dom_filter:
        not_simply_dominated, not_cumu_dominated = dom_filter(action_features, len_after_states=num_states)  # domtools.
        action_features = action_features[not_cumu_dominated]
        map_back_vector = np.nonzero(not_cumu_dominated)[0]
        # if rollout_cumu_dom_filter:
        #     available_after_states = available_after_states[not_simply_dominated]
        #     action_features = action_features[not_simply_dominated]
        # elif rollout_dom_filter:
        #     available_after_states = available_after_states[not_cumu_dominated]
        #     action_features = action_features[not_cumu_dominated]
    else:
        raise ValueError("Currently only implemented with cumu_dom_filter")
    utilities = action_features.dot(np.ascontiguousarray(policy_weights))
    move_index = np.argmax(utilities)
    move = available_after_states[map_back_vector[move_index]]
    return move

