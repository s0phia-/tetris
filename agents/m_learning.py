import numpy as np
from domtools import dom_filter as dominance_filter
from tetris import tetromino
from stew import StewMultinomialLogit, ChoiceSetData
from stew.utils import create_diff_matrix, create_ridge_matrix
from tetris.state import State
from numba import njit, float64
from scipy.stats import binom_test
import time


class MLearning:
    """
    M-learning, tailored to application in Tetris.
    """
    def __init__(self,
                 name,
                 regularization,
                 dom_filter,
                 cumu_dom_filter,
                 rollout_dom_filter,
                 rollout_cumu_dom_filter,
                 lambda_min,
                 lambda_max,
                 num_lambdas,
                 fixed_lambda,
                 gamma,
                 rollout_length,
                 number_of_rollouts_per_child,
                 learn_every_step_until,
                 max_batch_size,
                 learn_periodicity,
                 increase_learn_periodicity,
                 learn_from_step_in_current_phase,
                 num_columns,
                 feature_directors=np.array([-1, -1, -1, -1, -1, -1, 1, -1], dtype=np.float64),
                 feature_type="bcts",
                 verbose=False,
                 verbose_stew=False):

        self.name = name
        # Tetris params
        self.num_columns = num_columns  # ...of the Tetris board
        self.feature_type = feature_type
        self.num_features = 8
        self.feature_names = np.array(['rows_with_holes', 'column_transitions', 'holes',
                                       'landing_height', 'cumulative_wells', 'row_transitions',
                                       'eroded', 'hole_depth'])  # Uses BCTS features.
        self.verbose = verbose
        self.max_choice_set_size = 34  # There are never more than 34 actions in Tetris
        self.generative_model = tetromino.Tetromino(self.feature_type, self.num_features, self.num_columns)

        # Algo params
        self.gamma = gamma
        self.rollout_length = rollout_length
        self.number_of_rollouts_per_child = number_of_rollouts_per_child
        self.num_total_rollouts = self.rollout_length * self.number_of_rollouts_per_child
        self.dom_filter = dom_filter
        self.cumu_dom_filter = cumu_dom_filter
        self.rollout_dom_filter = rollout_dom_filter
        self.rollout_cumu_dom_filter = rollout_cumu_dom_filter

        # Algo init
        # self.policy_weights = np.random.normal(loc=0.0, scale=0.1, size=self.num_features)
        self.policy_weights = np.ones(self.num_features, dtype=np.float64)
        self.feature_directors = feature_directors
        self.feature_order = np.arange(self.num_features)
        self.step = 0
        self.step_in_current_phase = 0

        # Data and model (regularization type)
        self.regularization = regularization
        assert(self.regularization in ["stew", "no_regularization", "ridge", "nonnegative", "ew", "ttb", "stew_fixed_lambda"])
        self.is_learning = self.regularization in ["stew", "ridge", "no_regularization", "nonnegative", "stew_fixed_lambda"]
        if self.regularization == "ridge":
            D = create_ridge_matrix(self.num_features)
        elif self.regularization in ["stew", "stew_fixed_lambda", "nonnegative"]:
            D = create_diff_matrix(self.num_features)
        else:
            D = None
        self.fixed_lambda = fixed_lambda
        self.model = StewMultinomialLogit(num_features=self.num_features, D=D, lambda_min=lambda_min,
                                          lambda_max=lambda_max, num_lambdas=num_lambdas, verbose=verbose_stew,
                                          nonnegative=self.regularization == "nonnegative", one_se_rule=False)
        self.mlogit_data = ChoiceSetData(num_features=self.num_features, max_choice_set_size=self.max_choice_set_size)

        # Algo batch size handling
        self.delete_oldest_data_point_every = 2
        self.learn_from_step_in_current_phase = learn_from_step_in_current_phase
        self.learn_periodicity = learn_periodicity
        self.increase_learn_periodicity = increase_learn_periodicity
        self.learn_every_step_until = learn_every_step_until
        self.max_batch_size = max_batch_size
        self.step_since_last_learned = 0

    def copy_current_policy_weights(self):
        return self.policy_weights.copy() * self.feature_directors.copy()

    def update_steps(self):
        self.step += 1
        self.step_in_current_phase += 1

    def choose_action(self, start_state, start_tetromino):
        return choose_action_using_rollouts(start_state, start_tetromino,
                                            self.rollout_length, self.generative_model, self.policy_weights,
                                            self.dom_filter, self.cumu_dom_filter, self.rollout_dom_filter, self.rollout_cumu_dom_filter,
                                            self.feature_directors, self.num_features, self.gamma,
                                            self.number_of_rollouts_per_child)

    def append_data(self, action_features, action_index):
        delete_oldest = (self.mlogit_data.current_number_of_choice_sets > self.max_batch_size
                         or (self.delete_oldest_data_point_every > 0
                             and self.step_in_current_phase % self.delete_oldest_data_point_every == 0
                             and self.step_in_current_phase > self.learn_from_step_in_current_phase))
        # if self.verbose:
        #     print("Will delete oldest" if delete_oldest else "Will NOT delete oldest")
        self.mlogit_data.push(features=action_features, choice_index=action_index, delete_oldest=delete_oldest)

    def learn(self, action_features, action_index):
        """
        Learns new policy weights from choice set data.
        """
        self.step_since_last_learned += 1
        if self.step_in_current_phase >= self.learn_from_step_in_current_phase \
                and (self.step_in_current_phase <= self.learn_every_step_until
                     or self.step_since_last_learned >= self.learn_periodicity):
            self.learn_periodicity += self.increase_learn_periodicity
            # print("self.learn_periodicity", self.learn_periodicity)
            # print("Started learning")
            learning_time_start = time.time()
            if self.regularization in ["no_regularization", "nonnegative"]:
                self.policy_weights = self.model.fit(data=self.mlogit_data.sample(), lam=0, standardize=False)
            elif self.regularization == "stew_fixed_lambda":
                self.policy_weights = self.model.fit(data=self.mlogit_data.sample(), lam=self.fixed_lambda, standardize=False)
            elif self.regularization in ["ridge", "stew"]:
                self.policy_weights, _ = self.model.cv_fit(data=self.mlogit_data.sample())
            self.policy_weights = np.ascontiguousarray(self.policy_weights)
            # print("Learning took: " + str(time.time() - learning_time_start) + " seconds.")
            self.step_since_last_learned = 0


class HierarchicalLearning(MLearning):
    def __init__(self,
                 phase_names,
                 max_length_phase,
                 regularization,
                 dom_filter_per_phase,
                 cumu_dom_filter_per_phase,
                 rollout_dom_filter_per_phase,
                 rollout_cumu_dom_filter_per_phase,
                 rollout_mechanism_per_phase,
                 lambda_min,
                 lambda_max,
                 num_lambdas,
                 fixed_lambda,
                 gamma_per_phase,
                 append_data_per_phase,
                 rollout_length,
                 number_of_rollouts_per_child,
                 learn_every_step_until,
                 max_batch_size,
                 learn_periodicity,
                 increase_learn_periodicity,
                 learn_from_step_in_current_phase,
                 num_columns,
                 feature_type="bcts",
                 verbose=False,
                 verbose_stew=False,
                 provide_directions=False):
        self.phase_names = phase_names
        self.num_phases = len(self.phase_names)
        self.current_phase_index = 0
        self.current_phase = self.phase_names[self.current_phase_index]
        self.step_in_current_phase = 0
        self.max_length_phase = max_length_phase
        self.switched_phase_in_step = []

        self.dom_filter_per_phase = dom_filter_per_phase
        self.cumu_dom_filter_per_phase = cumu_dom_filter_per_phase
        self.rollout_dom_filter_per_phase = rollout_dom_filter_per_phase
        self.rollout_cumu_dom_filter_per_phase = rollout_cumu_dom_filter_per_phase
        self.gamma_per_phase = gamma_per_phase
        self.append_data_per_phase = append_data_per_phase
        self._append_data = self.append_data_per_phase[0]
        self.rollout_mechanism_per_phase = rollout_mechanism_per_phase
        # self.rollout_mechanism = self.determine_rollout_mechanism()
        self.rollout_mechanism = self.rollout_mechanism_per_phase[0]

        self.provide_directions = provide_directions

        if self.current_phase == "learn_weights" and feature_type == "bcts":
            if self.provide_directions:
                self.feature_directors = np.array([-1, -1, -1, -1, -1, -1, 1, -1], dtype=np.float64)
            else:
                self.feature_directors = (np.random.binomial(1, 0.5, 8) - 0.5) * 2
        elif self.current_phase == "learn_directions" and feature_type == "bcts":
            # TODO: random directions or all 1 here? (np.random.binomial(1, 0.5, 8) - 0.5) * 2
            # self.feature_directors = (np.random.binomial(1, 0.5, 8) - 0.5) * 2
            self.feature_directors = np.ones(8, dtype=np.float64)
        else:
            raise ValueError("Only bcts features are implemented.")

        self.check_arguments(regularization)

        super().__init__("hierarchical_learning", regularization, dom_filter_per_phase[0], cumu_dom_filter_per_phase[0], rollout_dom_filter_per_phase[0],
                         rollout_cumu_dom_filter_per_phase[0], lambda_min, lambda_max, num_lambdas, fixed_lambda, gamma_per_phase[0], rollout_length,
                         number_of_rollouts_per_child, learn_every_step_until, max_batch_size, learn_periodicity,
                         increase_learn_periodicity, learn_from_step_in_current_phase, num_columns, self.feature_directors, feature_type,
                         verbose, verbose_stew)

        self.positive_direction_counts = np.zeros(self.num_features)
        self.meaningful_comparisons = np.zeros(self.num_features)
        self.learned_directions = np.zeros(self.num_features)

    def check_arguments(self, regularization):
        assert regularization in ["no_regularization", "nonnegative", "ridge", "stew", "stew_fixed_lambda"]
        assert np.all(np.isin(np.array(self.phase_names), np.array(["learn_directions", "learn_order", "learn_weights"])))
        assert len(self.dom_filter_per_phase) == self.num_phases
        assert len(self.cumu_dom_filter_per_phase) == self.num_phases
        assert len(self.rollout_dom_filter_per_phase) == self.num_phases
        assert len(self.rollout_cumu_dom_filter_per_phase) == self.num_phases
        assert len(self.rollout_mechanism_per_phase) == self.num_phases
        assert len(self.gamma_per_phase) == self.num_phases
        assert len(self.append_data_per_phase) == self.num_phases

    def copy_current_policy_weights(self):
        if self.current_phase == "learn_directions":
            return self.policy_weights.copy() * self.learned_directions.copy()
        elif self.current_phase == "learn_weights":
            return self.policy_weights.copy() * self.feature_directors.copy()

    def copy_current_feature_directors(self):
        if self.current_phase == "learn_directions":
            return self.learned_directions.copy()
        elif self.current_phase == "learn_weights":
            return self.feature_directors.copy()

    def append_data(self, action_features, action_index):
        if self._append_data:
            super().append_data(action_features, action_index)
        # elif self.current_phase == "learn_weights":
        #     super().append_data(action_features, action_index)

    # def determine_rollout_mechanism(self):
    #     """
    #     Deterministic mapping from phase to rollout mechanism. Later this can be made a parameter.
    #     :return: string, the rollout mechanism
    #     """
    #     if self.current_phase == "learn_directions":
    #         return "greedy_if_reward_else_random"
    #     elif self.current_phase == "learn_weights":
    #         return "max_util"

    def learn(self, action_features, action_index):
        self.append_data(action_features, action_index)
        if self.current_phase in ["learn_directions", "learn_order"]:  # self.phase == "learn_directions"
            chosen_action_features = action_features[action_index]
            remaining_action_features = np.delete(arr=action_features, obj=action_index, axis=0)
            feature_differences = np.sign(chosen_action_features - remaining_action_features)
            direction_counts = np.sign(np.sum(feature_differences, axis=0))
            self.positive_direction_counts += np.maximum(direction_counts, 0)
            self.meaningful_comparisons += np.abs(direction_counts)
        elif self.current_phase in ["learn_weights"]:
            super().learn(action_features, action_index)
        switched_phase = self.check_phase()
        return switched_phase

    def check_phase(self):
        switched_phase = False
        # self.step_in_current_phase += 1
        if self.current_phase == "learn_directions":
            # if self.step_in_current_phase >= 5:
            unidentified_directions = np.where(self.learned_directions == 0.)[0]
            for feature_ix in range(len(unidentified_directions)):
                feature = unidentified_directions[feature_ix]
                p_value = binom_test(x=self.positive_direction_counts[feature],
                                     n=self.meaningful_comparisons[feature],
                                     p=0.5, alternative="two-sided")
                if self.verbose:
                    print("Feature ", feature, " has ", self.positive_direction_counts[feature], "/", self.meaningful_comparisons[feature],
                          "positive effective comparisons. P-value: ", np.round(p_value, 4))
                if p_value < 0.05:
                    self.learned_directions[feature] = np.sign(self.positive_direction_counts[feature] / self.meaningful_comparisons[feature] - 0.5)
                    # if self.verbose:
                    print("Feature ", feature, " has been decided to be: ", self.learned_directions[feature])
            if np.all(self.learned_directions != 0.):
                # if self.verbose:
                print("All directions have been identified. They are:")
                print(self.learned_directions)
                self.switch_phase()
                switched_phase = True
        # elif self.phase == "learn_order":
        #     temp_ratios = self.positive_direction_counts / self.meaningful_comparisons
        #     self.learned_order = np.argsort(-temp_ratios)
        #     sorted_ratios = temp_ratios[self.learned_order]
        #     p_values = np.zeros(self.num_features-1)
        #     for ix in range(len(sorted_ratios)-1):
        #         stat, p_value = proportions_ztest(self.positive_direction_counts[ix:ix+2], self.meaningful_comparisons[ix:ix+2])
        #         if self.verbose:
        #             print("Features ", ix, " and ", ix + 1, " have proportions of ", sorted_ratios[ix], " and ", sorted_ratios[ix+1])
        #             print("The proportion_z_test shows a p-value of: ", np.round(p_value, 4))
        #         p_values[ix] = p_value
        #     if np.all(p_values < 0.1):
        #         print("The complete order has been identified. It is:")
        #         print(self.learned_order)
        #         print("The p-values are:")
        #         print(p_values)
        #         print("Switching to 'learn_weights' phase.")
        #         self.switch_phase()
        #         switched_phase = True
        elif self.current_phase == "learn_weights":
            pass
        else:
            raise ValueError("self.phase wrongly specified!")
        if self.current_phase_index < (self.num_phases - 1) and self.step_in_current_phase > self.max_length_phase:
            print("SWITCHING PHASE because max_length_phase of " + str(self.max_length_phase) + " has been reached!")
            self.switch_phase()
            switched_phase = True
        return switched_phase

    def switch_phase(self):
        if self.num_phases - 1 > self.current_phase_index:
            self.switched_phase_in_step.append(self.step)
            old_phase = self.current_phase
            self.step_in_current_phase = 0
            self.current_phase_index += 1
            self.current_phase = self.phase_names[self.current_phase_index]

            # self.gamma = self.gamma_phases[self.phase_ix]
            # self.rollout_length = self.rollout_length_phases[self.phase_ix]
            # self.rollout_action_selection = self.rollout_action_selection_phases[self.phase_ix]
            # self.max_length_phase = self.max_length_phases[self.phase_ix]
            self.dom_filter = self.dom_filter_per_phase[self.current_phase_index]
            self.cumu_dom_filter = self.cumu_dom_filter_per_phase[self.current_phase_index]
            self.rollout_dom_filter = self.rollout_dom_filter_per_phase[self.current_phase_index]
            self.rollout_cumu_dom_filter = self.rollout_cumu_dom_filter_per_phase[self.current_phase_index]
            self.rollout_mechanism = self.rollout_mechanism_per_phase[self.current_phase_index]
            self.gamma = self.gamma_per_phase[self.current_phase_index]
            self._append_data = self.append_data_per_phase[self.current_phase_index]
            # self.filter_best = self.filter_best_phases[self.phase_ix]
            # self.ols = self.ols_phases[self.phase_ix]
            # self.delete_oldest_data_point_every = self.delete_oldest_data_point_every_phases[self.phase_ix]
            # self.learn_from_step_in_current_phase = self.learn_from_step_phases[self.phase_ix]
            # self.number_of_rollouts_per_child = self.number_of_rollouts_per_child_phases[self.phase_ix]
            # self.num_total_rollouts = self.rollout_length * self.number_of_rollouts_per_child
            print("Switching to phase: ", self.current_phase)
            print("self.dom_filter = ", self.dom_filter)
            print("self.cumu_dom_filter = ", self.cumu_dom_filter)
            print("self.rollout_dom_filter = ", self.rollout_dom_filter)
            print("self.rollout_cumu_dom_filter = ", self.rollout_cumu_dom_filter)
            print("self.rollout_mechanism = ", self.rollout_mechanism)
            print("self.gamma = ", self.gamma)
            print("self._append_data = ", self._append_data)

            if old_phase == "learn_directions":  # self.phase == "learn_order":  # coming from "learn_directions"
                print("The learned directions are: ", self.learned_directions)
                self.feature_directors = np.sign(self.positive_direction_counts / self.meaningful_comparisons - 0.5)
                print("Feature directors will be: ", self.feature_directors)
                if self.current_phase == "learn_order":
                    self.positive_direction_counts = flip_positive_direction_counts(self.positive_direction_counts,
                                                                                    self.meaningful_comparisons,
                                                                                    self.feature_directors)
                # Update existing training data with new feature_directions
                # TODO: reintroduce if data from learn directions is stored.
                # self.mlogit_data.data[:, 2:] = self.mlogit_data.data[:, 2:] * self.feature_directors
            elif old_phase == "learn_order":  #self.phase == "learn_weights":  # coming from "learn_order"
                # Feature variance business
                print("Estimating feature variance from", len(self.mlogit_data.data), "data points")
                std_deviations = np.std(self.mlogit_data.data[:, 2:], axis=0)
                self.feature_stds = std_deviations
                print("The standard deviations are ", self.feature_stds)
                print("... for features", self.feature_names)
                self.mlogit_data.delete_data()

                # Order
                print("The original order was", self.feature_names)
                self.feature_order = self.feature_order[self.learned_order]
                self.feature_names = self.feature_names[self.learned_order]
                self.feature_stds = self.feature_stds[self.learned_order]
                print("The new order is", self.feature_names)
                self.feature_directors = self.feature_directors[self.learned_order]
                print("...and accordingly, the new directions are: ", self.feature_directors)

    def choose_action(self, start_state, start_tetromino):
        return choose_action_using_rollouts(start_state, start_tetromino, self.rollout_mechanism,
                                            self.rollout_length, self.generative_model, self.policy_weights,
                                            self.dom_filter, self.cumu_dom_filter, self.rollout_dom_filter,
                                            self.rollout_cumu_dom_filter,
                                            self.feature_directors,
                                            self.num_features, self.gamma,
                                            self.number_of_rollouts_per_child,
                                            self.learned_directions)
        # if self.rollout_mechanism == "max_util":
        #     super().choose_action(start_state, start_tetromino)
        # elif self.rollout_mechanism == "greedy_if_reward_else_random":
        #     return choose_action_using_rollouts(start_state, start_tetromino, self.rollout_mechanism,
        #                                         self.rollout_length, self.generative_model, self.policy_weights,
        #                                         self.dom_filter, self.cumu_dom_filter, self.rollout_dom_filter,
        #                                         self.rollout_cumu_dom_filter,
        #                                         self.feature_directors,
        #                                         self.num_features, self.gamma,
        #                                         self.number_of_rollouts_per_child,
        #                                         self.learned_directions)
        # elif self.rollout_mechanism == "greedy_if_reward_else_max_util_from_learned_directions":
        #     return choose_action_using_rollouts(start_state, start_tetromino, self.rollout_mechanism,
        #                                         self.rollout_length, self.generative_model, self.policy_weights,
        #                                         self.dom_filter, self.cumu_dom_filter, self.rollout_dom_filter,
        #                                         self.rollout_cumu_dom_filter,
        #                                         self.feature_directors,   #### IMPORTANT CHANGE
        #                                         self.num_features, self.gamma,
        #                                         self.number_of_rollouts_per_child,
        #                                         self.learned_directions)


# @njit(cache=False)
def choose_action_using_rollouts(start_state, start_tetromino, rollout_mechanism,
                                 rollout_length, generative_model, policy_weights,
                                 dom_filter, cumu_dom_filter, rollout_dom_filter, rollout_cumu_dom_filter,
                                 feature_directors, num_features, gamma, number_of_rollouts_per_child,
                                 learned_directions):
    children_states = start_tetromino.get_after_states(start_state)
    num_children = len(children_states)
    if num_children == 0:
        # Game over!
        return (State(np.zeros((1, 1), dtype=np.bool_), np.zeros(1, dtype=np.int64),
                      np.array([0], dtype=np.int64), np.array([0], dtype=np.int64),
                      0.0, 1, "bcts", True, False),  # dummy state
                0,                                   # dummy child_index
                np.zeros((2, 2)))                    # dummy action_features

    action_features = np.zeros((num_children, num_features), dtype=np.float_)
    for ix in range(num_children):
        action_features[ix] = children_states[ix].get_features_and_direct(feature_directors, False)  # , order_by=self.feature_order
    if dom_filter or cumu_dom_filter:
        not_simply_dominated, not_cumu_dominated = dominance_filter(action_features, len_after_states=num_children)

    child_total_values = np.zeros(num_children)
    for child in range(num_children):
        do_rollout = False
        if cumu_dom_filter:
            if not_cumu_dominated[child]:
                do_rollout = True
        elif dom_filter:
            if not_simply_dominated[child]:
                do_rollout = True
        else:
            do_rollout = True

        if do_rollout:
            for rollout_ix in range(number_of_rollouts_per_child):
                child_total_values[child] += roll_out(children_states[child], rollout_length, rollout_mechanism,
                                                      generative_model, policy_weights,
                                                      rollout_dom_filter, rollout_cumu_dom_filter,
                                                      feature_directors, num_features, gamma, learned_directions)
        else:
            child_total_values[child] = -np.inf

    max_value = np.max(child_total_values)
    max_value_indices = np.where(child_total_values == max_value)[0]
    child_index = np.random.choice(max_value_indices)
    return children_states[child_index], child_index, action_features


# @njit(cache=False)
def roll_out(start_state, rollout_length, rollout_mechanism,
             generative_model, policy_weights,
             rollout_dom_filter, rollout_cumu_dom_filter,
             feature_directors, num_features, gamma, learned_directions):
    value_estimate = start_state.n_cleared_lines
    state_tmp = start_state
    count = 1
    while not state_tmp.terminal_state and count <= rollout_length:
        generative_model.next_tetromino()
        available_after_states = generative_model.get_after_states(state_tmp)
        if len(available_after_states) == 0:
            # Game over!
            return value_estimate
        if rollout_mechanism == "max_util":
            state_tmp = choose_max_util_action_in_rollout(
                available_after_states, policy_weights,
                rollout_dom_filter, rollout_cumu_dom_filter,
                feature_directors, num_features)
        elif rollout_mechanism == "greedy_if_reward_else_random":
            state_tmp = choose_greedy_if_reward_else_random_action_in_rollout(
                available_after_states, policy_weights,
                rollout_dom_filter, rollout_cumu_dom_filter,
                feature_directors, num_features)
        elif rollout_mechanism == "greedy_if_reward_else_max_util_from_learned_directions":
            state_tmp = choose_greedy_if_reward_else_max_util_from_learned_directions_action_in_rollout(
                available_after_states, policy_weights,
                rollout_dom_filter, rollout_cumu_dom_filter,
                feature_directors, num_features, learned_directions)

        value_estimate += gamma ** count * state_tmp.n_cleared_lines
        count += 1
    return value_estimate


# @njit(cache=False)
def choose_max_util_action_in_rollout(available_after_states, policy_weights,
                                      rollout_dom_filter, rollout_cumu_dom_filter,
                                      feature_directors, num_features):
    num_states = len(available_after_states)
    action_features = np.zeros((num_states, num_features))
    for ix, after_state in enumerate(available_after_states):
        action_features[ix] = after_state.get_features_and_direct(feature_directors, False)  # , order_by=self.feature_order
    if rollout_dom_filter or rollout_cumu_dom_filter:
        not_simply_dominated, not_cumu_dominated = dominance_filter(action_features, len_after_states=num_states)  # domtools.
        if rollout_cumu_dom_filter:
            action_features = action_features[not_cumu_dominated]
            map_back_vector = np.nonzero(not_cumu_dominated)[0]
        elif rollout_dom_filter:
            action_features = action_features[not_simply_dominated]
            map_back_vector = np.nonzero(not_simply_dominated)[0]
    else:
        map_back_vector = np.arange(num_states)
    utilities = action_features.dot(policy_weights)
    move_index = np.random.choice(map_back_vector[utilities == np.max(utilities)])
    # move_index = np.argmax(utilities)
    move = available_after_states[move_index]
    return move


# @njit(cache=False)
def choose_greedy_if_reward_else_random_action_in_rollout(available_after_states, policy_weights,
                                                          rollout_dom_filter, rollout_cumu_dom_filter,
                                                          feature_directors, num_features):
    num_states = len(available_after_states)
    if rollout_dom_filter or rollout_cumu_dom_filter:
        action_features = np.zeros((num_states, num_features))
        for ix, after_state in enumerate(available_after_states):
            action_features[ix] = after_state.get_features_and_direct(feature_directors, False)  # , order_by=self.feature_order
        not_simply_dominated, not_cumu_dominated = dominance_filter(action_features, len_after_states=num_states)  # domtools.
        if rollout_cumu_dom_filter:
            available_after_states = [s for (s, d) in zip(available_after_states, not_cumu_dominated) if d]
            # map_back_vector = np.nonzero(not_cumu_dominated)[0]
            # available_after_states = [available_after_states[i] for i in map_back_vector]
        elif rollout_dom_filter:
            available_after_states = [s for (s, d) in zip(available_after_states, not_simply_dominated) if d]
            # map_back_vector = np.nonzero(not_simply_dominated)[0]
            # available_after_states = available_after_states[map_back_vector]
        num_states = len(available_after_states)
    # else:
    #     map_back_vector = np.arange(num_states)

    rewards = np.zeros(num_states)
    max_reward = 0
    for ix, after_state in enumerate(available_after_states):
        reward_of_after_state = after_state.n_cleared_lines
        if reward_of_after_state > 0:
            rewards[ix] = after_state.n_cleared_lines
            if reward_of_after_state > max_reward:
                max_reward = reward_of_after_state
    if max_reward > 0:
        max_reward_indeces = np.where(rewards == max_reward)[0]
        move_index = np.random.choice(max_reward_indeces)
        # move = np.random.choice([available_after_states[i] for i in max_reward_indeces])
    else:
        move_index = np.random.choice(num_states)
        # move = np.random.choice(available_after_states)
    move = available_after_states[move_index]
    return move


# @njit(cache=False)
def choose_greedy_if_reward_else_max_util_from_learned_directions_action_in_rollout(
        available_after_states, policy_weights,
        rollout_dom_filter, rollout_cumu_dom_filter,
        feature_directors,
        num_features,
        learned_directions):
    num_states = len(available_after_states)
    action_features = np.zeros((num_states, num_features))
    for ix, after_state in enumerate(available_after_states):
        action_features[ix] = after_state.get_features_and_direct(feature_directors, False)  # , order_by=self.feature_order

    if rollout_dom_filter or rollout_cumu_dom_filter:
        not_simply_dominated, not_cumu_dominated = dominance_filter(action_features, len_after_states=num_states)  # domtools.
        if rollout_cumu_dom_filter:
            available_after_states = [s for (s, d) in zip(available_after_states, not_cumu_dominated) if d]
            # map_back_vector = np.nonzero(not_cumu_dominated)[0]
            # available_after_states = [available_after_states[i] for i in map_back_vector]
        elif rollout_dom_filter:
            available_after_states = [s for (s, d) in zip(available_after_states, not_simply_dominated) if d]
            # map_back_vector = np.nonzero(not_simply_dominated)[0]
            # available_after_states = available_after_states[map_back_vector]
        num_states = len(available_after_states)
    # else:
    #     map_back_vector = np.arange(num_states)

    rewards = np.zeros(num_states)
    max_reward = 0
    for ix, after_state in enumerate(available_after_states):
        reward_of_after_state = after_state.n_cleared_lines
        if reward_of_after_state > 0:
            rewards[ix] = after_state.n_cleared_lines
            if reward_of_after_state > max_reward:
                max_reward = reward_of_after_state
    if max_reward > 0:
        # max_reward_indeces = rewards == max_reward
        # available_after_states = [s for (s, d) in zip(available_after_states, not_simply_dominated) if d]
        max_reward_indeces = np.where(rewards == max_reward)[0]
        available_after_states = [available_after_states[i] for i in max_reward_indeces]
        action_features = action_features[max_reward_indeces]
        num_states = len(available_after_states)
    utilities = action_features.dot(policy_weights * learned_directions)
    # utilities == np.max(utilities)
    move_index = np.random.choice(np.arange(num_states)[utilities == np.max(utilities)])
    # move_index = np.argmax(utilities)
    move = available_after_states[move_index]
    return move


# @njit
def flip_positive_direction_counts(positive_direction_counts, meaningful_comparisons, feature_directors):
    for ix in range(len(positive_direction_counts)):
        if feature_directors[ix] == -1.:
            positive_direction_counts[ix] = meaningful_comparisons[ix] - positive_direction_counts[ix]
    return positive_direction_counts




