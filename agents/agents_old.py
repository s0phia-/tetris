import numpy as np

import stew
import domtools

from tetris import mcts, tetromino_old
from numba import njit
from scipy.stats import binom_test
from statsmodels.stats.proportion import proportions_ztest


class RolloutActorCritic:
    # No bootstrapping
    def __init__(self, rollout_length, temperature, filter_in_rollout, num_features, feature_type, policy_lambda, policy_alpha, gamma,
                 avg_expands_per_children, ucb_sampling, num_columns, cp, dom_filter, cumu_dom_filter, add_reward, target_update,
                 mlogit_learning, max_sample_size, mlogit_data, memory_size, learn_from_step,
                 lambda_min=-8.0, lambda_max=4.0, num_lambdas=50, verbose=False):
        self.num_features = num_features
        self.feature_type = feature_type
        if self.feature_type == "bcts":
            self.standardize_features = True
            self.feature_names = np.array(['rows_with_holes', 'column_transitions', 'holes', 'landing_height', 'cumulative_wells',
                                  'row_transitions', 'eroded', 'hole_depth'])
            self.policy_weights = np.array([-1, -1, -1, -1, -1, -1, 1, -1]) / 10
        elif self.feature_type == "standardized_bcts":
            self.standardize_features = False
            self.policy_weights = np.array([1, -1, -1, -1, -1, -1, -1, -1]) / 10
        self.temperature = temperature
        self.rollout_length = rollout_length
        self.filter_in_rollout = filter_in_rollout
        self.verbose = verbose
        self.stew = stew.StewMultinomialLogit(num_features=self.num_features, lambda_min=lambda_min, lambda_max=lambda_max,
                                              num_lambdas=num_lambdas, verbose=self.verbose)
        self.learn_from_step = learn_from_step
        self.feature_directors = np.ones(num_features)
        self.feature_directors *= np.sign(self.policy_weights)
        self.policy_weights *= np.sign(self.policy_weights)
        self.step = 0
        self.max_choice_set_size = 35
        self.delete_every = 2
        # self.action_selection_method = "rollout_with_prob"
        self.action_selection_method = "rollout"
        # TODO: add directions

        self.gamma = gamma
        self.policy_lambda = policy_lambda
        self.policy_alpha = policy_alpha
        self.policy_eligibility_trace = np.zeros(self.num_features)
        self.add_reward = add_reward
        self.avg_expands_per_children = avg_expands_per_children
        self.num_columns = num_columns
        self.cp = cp
        self.dom_filter = dom_filter
        self.cumu_dom_filter = cumu_dom_filter
        self.ucb_sampling = ucb_sampling

        self.target_update = target_update  # only here for compatibility reasons

        self.tetrominos = [tetromino_old.Straight(self.feature_type, self.num_features, self.num_columns),
                           tetromino_old.RCorner(self.feature_type, self.num_features, self.num_columns),
                           tetromino_old.LCorner(self.feature_type, self.num_features, self.num_columns),
                           tetromino_old.Square(self.feature_type, self.num_features, self.num_columns),
                           tetromino_old.SnakeR(self.feature_type, self.num_features, self.num_columns),
                           tetromino_old.SnakeL(self.feature_type, self.num_features, self.num_columns),
                           tetromino_old.T(self.feature_type, self.num_features, self.num_columns)]
        self.tetromino_sampler = tetromino_old.TetrominoSampler(self.tetrominos)

        # Experience replay
        self.mlogit_learning = mlogit_learning
        if self.mlogit_learning:
            if mlogit_data is None:
                self.mlogit_data = MlogitData(num_features=self.num_features, capacity=self.memory_size)
            else:
                self.mlogit_data = mlogit_data

    def reset_agent(self):
        self.policy_eligibility_trace = np.zeros(self.num_features)
        self.step = 0

    def choose_action(self, environment, start_state, start_tetromino):
        if self.action_selection_method == "ucb_sampling":
            root = mcts.NodeRAC(state=start_state, tetromino=start_tetromino, environment=environment,
                                dom_filter=self.dom_filter, cumu_dom_filter=self.cumu_dom_filter,
                                feature_directors=self.feature_directors, parent=mcts.DummyNode(), cp=self.cp)
            root.expand()
            action_features = root.child_features[root.tetromino_name]
            root.child_priors[root.tetromino_name] = compute_action_probabilities(action_features, self.policy_weights,
                                                                                  self.temperature)
            num_children = root.num_children
            # if num_children > 1:
            for ix in range(num_children):
                leaf = root.children[root.tetromino_name][ix]
                value_estimate = self.roll_out(leaf.state)
                # child_priors, value_estimate = self.mlp.evaluate(leaf.game_state)
                leaf.backup(value_estimate)
            child_rollout_values = root.child_Q()
            self.avg_rollout_value = np.median(child_rollout_values)
            # Store probabilities for learning before filtering actions
            self.probabilities = compute_action_probabilities(action_features, self.policy_weights, self.temperature)
            map_back_vector = root.filter()
            num_children_filtered = root.num_children
            # if num_children_filtered > 1:
            num_expands_tmp = num_children_filtered * self.avg_expands_per_children
            for _ in range(num_expands_tmp):
                leaf = root.best_child()
                value_estimate = self.roll_out(leaf.state)
                # for ix in range(root.num_children):
                #     print(root.children[root.tetromino_name][ix].move_index)
                leaf.backup(value_estimate)
            # IDEA: re-weight Q and U, s.t. action selection probs follow Zipf's law
            # freqs = np.sort(root.child_number_visits[root.tetromino_name])[::-1]
            # print(root.child_number_visits)
            # print(root.child_Q())
            child_rollout_values = root.child_Q()
            # self.avg_rollout_value = np.median(root.child_Q())
            child_index = np.argmax(child_rollout_values)
            before_filter_index = map_back_vector[child_index]  # Needed for probabilities in gradient in learn()
            root.children[root.tetromino_name][child_index].store_Q_estimate()
            return root.children[root.tetromino_name][child_index].state, before_filter_index
        elif self.action_selection_method == "rollout_with_prob":
            all_children_states = start_tetromino.get_after_states(current_state=start_state)
            children_states = np.array([child for child in all_children_states if not child.terminal_state])
            if len(children_states) == 0:
                # Game over!
                return all_children_states[0], 0, None
            num_children = len(children_states)
            action_features = np.zeros((num_children, self.num_features), dtype=np.float_)
            child_total_values = np.zeros(num_children)
            for ix in range(num_children):
                action_features[ix] = children_states[ix].get_features(direct_by=self.feature_directors)
                child_total_values[ix] = self.roll_out(children_states[ix])
            self.avg_rollout_value = np.median(child_total_values)
            # Store probabilities for learning before filtering actions
            self.probabilities = compute_action_probabilities(action_features, self.policy_weights, self.temperature)
            not_simply_dominated, not_cumu_dominated = domtools.dom_filter(action_features, len_after_states=num_children)
            if self.cumu_dom_filter:
                children_states = children_states[not_cumu_dominated]
                # filtered_features = filtered_features[not_cumu_dominated]
                child_total_values = child_total_values[not_cumu_dominated]
                map_back_vector = np.nonzero(not_cumu_dominated)[0]
            else:  # Only simple dom
                children_states = children_states[not_simply_dominated]
                # filtered_features = filtered_features[not_simply_dominated]
                child_total_values = child_total_values[not_simply_dominated]
                map_back_vector = np.nonzero(not_simply_dominated)[0]
            num_children_filtered = len(children_states)
            # num_expands_tmp = num_children_filtered * self.avg_expands_per_children
            for child in range(num_children_filtered):
                for rollout in range(self.avg_expands_per_children):
                    child_total_values[child] += self.roll_out(children_states[child])
            child_index = np.argmax(child_total_values)
            children_states[child_index].value_estimate = child_total_values[child_index]
            before_filter_index = map_back_vector[child_index]  # Needed for probabilities in gradient in learn()
            return children_states[child_index], before_filter_index, action_features
        elif self.action_selection_method == "rollout":
            all_children_states = start_tetromino.get_after_states(current_state=start_state)
            children_states = np.array([child for child in all_children_states if not child.terminal_state])
            if len(children_states) == 0:
                # Game over!
                return all_children_states[0], 0, None
            num_children = len(children_states)
            action_features = np.zeros((num_children, self.num_features), dtype=np.float_)
            for ix in range(num_children):
                action_features[ix] = children_states[ix].get_features(direct_by=self.feature_directors)
            not_simply_dominated, not_cumu_dominated = domtools.dom_filter(action_features, len_after_states=num_children)
            if self.cumu_dom_filter:
                children_states = children_states[not_cumu_dominated]
                map_back_vector = np.nonzero(not_cumu_dominated)[0]
            else:  # Only simple dom
                children_states = children_states[not_simply_dominated]
                map_back_vector = np.nonzero(not_simply_dominated)[0]
            num_children_filtered = len(children_states)
            child_total_values = np.zeros(num_children_filtered)
            for child in range(num_children_filtered):
                for rollout in range(self.avg_expands_per_children):
                    child_total_values[child] += self.roll_out(children_states[child])
            child_index = np.argmax(child_total_values)
            children_states[child_index].value_estimate = child_total_values[child_index]
            before_filter_index = map_back_vector[child_index]  # Needed for probabilities in gradient in learn()
            return children_states[child_index], before_filter_index, action_features

    def choose_action_test(self, start_state, start_tetromino):
        all_available_after_states = start_tetromino.get_after_states(current_state=start_state)
        available_after_states = np.array([child for child in all_available_after_states if not child.terminal_state])
        if len(available_after_states) == 0:
            # Game over!
            return all_available_after_states[0], 0
        num_states = len(available_after_states)
        action_features = np.zeros((num_states, self.num_features))
        for ix, after_state in enumerate(available_after_states):
            action_features[ix] = after_state.get_features(direct_by=self.feature_directors)
        utilities = action_features.dot(self.policy_weights)
        max_indices = np.where(utilities == np.max(utilities))[0]
        move_index = np.random.choice(max_indices)
        move = available_after_states[move_index]
        return move, move_index

    def choose_rollout_action(self, available_after_states):
        num_states = len(available_after_states)
        action_features = np.zeros((num_states, self.num_features))
        for ix, after_state in enumerate(available_after_states):
            action_features[ix] = after_state.get_features(direct_by=self.feature_directors)
        if self.filter_in_rollout:
            not_simply_dominated, not_cumu_dominated = domtools.dom_filter(action_features, len_after_states=num_states)
            if self.cumu_dom_filter:
                available_after_states = available_after_states[not_cumu_dominated]
                action_features = action_features[not_cumu_dominated]
            else:  # Only simple dom
                available_after_states = available_after_states[not_simply_dominated]
                action_features = action_features[not_simply_dominated]
        utilities = action_features.dot(self.policy_weights)
        move = available_after_states[np.argmax(utilities)]
        return move

    def roll_out(self, start_state):
        value_estimate = start_state.reward
        state_tmp = start_state
        count = 1
        while not state_tmp.terminal_state and count <= self.rollout_length:
            tetromino_tmp = self.tetromino_sampler.next_tetromino()
            all_available_after_states = tetromino_tmp.get_after_states(state_tmp)
            non_terminal_after_states = np.array([child for child in all_available_after_states if not child.terminal_state])
            if len(non_terminal_after_states) == 0:
                # Game over!
                return value_estimate
            state_tmp = self.choose_rollout_action(non_terminal_after_states)
            value_estimate += self.gamma ** count * state_tmp.reward
            count += 1
        return value_estimate

    def choose_action_store_features(self, start_state, start_tetromino):
        all_available_after_states = start_tetromino.get_after_states(current_state=start_state)
        available_after_states = np.array([child for child in all_available_after_states if not child.terminal_state])
        if len(available_after_states) == 0:
            # Game over!
            return all_available_after_states[0], 0, None
        num_states = len(available_after_states)
        action_features = np.zeros((num_states, self.num_features))
        for ix, after_state in enumerate(available_after_states):
            action_features[ix] = after_state.get_features(direct_by=self.feature_directors)
        utilities = action_features.dot(self.policy_weights)
        max_indices = np.where(utilities == np.max(utilities))[0]
        # probabilities = compute_action_probabilities(action_features, self.policy_weights, temperature=0.001)
        # move_index = int(np.random.choice(a=np.arange(num_states), size=1, p=probabilities))
        move_index = np.random.choice(max_indices)
        move = available_after_states[move_index]
        return move, move_index, action_features

    # def push_data(self, action_features, action_index):
    #     self.mlogit_data.push(features=action_features, choice_index=action_index, delete_oldest=False)

    def learn(self, action_features, action_index):
        if self.mlogit_learning:
            # delete_oldest = self.step >= self.memory_size and self.step % self.delete_every == 0
            delete_oldest = False
            self.mlogit_data.push(features=action_features, choice_index=action_index, delete_oldest=delete_oldest)
            if self.step >= self.learn_from_step:
                data_tmp = self.mlogit_data.sample()
                self.policy_weights, _ = self.stew.cv_fit(data=data_tmp, standardize=self.standardize_features)
        # else:
        #     # REINFORCE style learning
        #     delta = after_state.value_estimate - self.avg_rollout_value
        #     # probabilities = compute_action_probabilities(action_features, self.policy_weights, self.temperature)
        #     policy_grad = grad_of_log_action_probabilities(action_features, self.probabilities, action_index)
        #     self.policy_eligibility_trace = self.gamma * self.policy_lambda * self.policy_eligibility_trace + policy_grad
        #     self.policy_weights += self.policy_alpha * delta * self.policy_eligibility_trace
        #     # Re-direct!
        #     self.feature_directors *= np.sign(self.policy_weights)
        #     self.policy_weights *= np.sign(self.policy_weights)
