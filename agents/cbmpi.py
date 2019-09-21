import numpy as np
from stew import StewMultinomialLogit, ChoiceSetData
from tetris import tetromino
from numba import njit
from sklearn.linear_model import LinearRegression
import gc
import cma
import time


class MultinomialLogisticRegression:
    pass
    # self.discrete_choice = discrete_choice
    # if self.discrete_choice:
    #     self.regularization = "no_regularization"
    #     self.model = StewMultinomialLogit(num_features=self.num_features)
    #     self.mlogit_data = ChoiceSetData(num_features=self.num_features, max_choice_set_size=self.max_choice_set_size)
    #
    # if self.discrete_choice:
    #     self.mlogit_data.delete_data()
    #     # stacked_features = np.concatenate([self.state_action_features, axis=0)
    #     # TODO: rewrite to use less copying / memory...
    #     for state_ix in range(self.N):
    #         state_act_feat_ix = self.state_action_features[state_ix]
    #         if state_act_feat_ix is not None:
    #             action_values = self.state_action_values[state_ix, :len(state_act_feat_ix)]
    #             self.mlogit_data.push(features=self.state_action_features[state_ix], choice_index=np.argmax(action_values), delete_oldest=False)
    #     if self.ols:
    #         self.policy_weights = self.model.fit(data=self.mlogit_data.sample(), lam=0, standardize=False)
    #     # else:
    #     #     self.policy_weights, _ = self.model.cv_fit(data=self.mlogit_data.sample(), standardize=self.standardize_features)

class CmaesClassifier:
    def __init__(self, feature_type, cmaes_var, min_iterations, seed=0):
        self.name = "cmaes"
        if feature_type == "bcts":
            self.num_features = 8
        else:
            raise ValueError("Only BCTS features are implemented!")
        self.policy_weights = np.random.normal(loc=0, scale=cmaes_var, size=self.num_features)
        self.n = int(self.num_features * 15)  # CMA-ES parameter ('popsize' in cma)
        self.min_iterations = min_iterations
        self.cmaes = cma.CMAEvolutionStrategy(
            np.random.normal(loc=0, scale=1, size=self.num_features),
            cmaes_var,
            inopts={'verb_disp': 0,
                    'verb_filenameprefix': "output/cmaesout" + str(seed),
                    'popsize': self.n})

    def fit(self, state_action_features, state_action_values, did_rollout, num_available_actions):
        # self.policy_approximator = cma.CMAEvolutionStrategy(np.random.normal(loc=0, scale=1, size=self.num_features), self.cmaes_var,
        #                                            inopts={'verb_disp': 0,
        #                                                    'verb_filenameprefix': "output/cmaesout" + str(self.seed),
        #                                                    'popsize': self.n})
        N = len(state_action_features)
        policy_weights = self.cmaes.optimize(lambda x: policy_loss_function(x,
                                                                            N,
                                                                            did_rollout,
                                                                            state_action_features,
                                                                            num_available_actions,
                                                                            state_action_values),
                                             min_iterations=self.min_iterations).result.xbest
        return policy_weights


class LinearFunction:
    def __init__(self, feature_type="bcts+rbf"):
        self.name = "linear"
        if feature_type == "bcts+rbf":
            self.num_value_features = 13
        self.value_weights = np.zeros(self.num_value_features + 1)  # +1 is for intercept
        self.lin_reg = LinearRegression(n_jobs=1, fit_intercept=True)  # n_jobs must be 1... otherwise clashes with multiprocessing.Pool
        self.lin_reg.coef_ = np.zeros(self.num_value_features)
        self.lin_reg.intercept_ = 0.

    def fit(self, state_features, state_values):
        self.lin_reg.fit(state_features, state_values)
        value_weights = np.hstack((self.lin_reg.intercept_, self.lin_reg.coef_))
        return value_weights


class RolloutHandler:
    def __init__(self, rollout_state_population, budget, rollout_length, rollouts_per_action, num_features, num_value_features,
                 rollout_set_size=None, gamma=1):   # ,
        self.rollout_state_population = rollout_state_population
        self.rollout_set = None  # use self.construct_rollout_set()
        self.budget = budget
        self.rollout_length = rollout_length
        self.rollouts_per_action = rollouts_per_action
        if rollout_set_size is None:
            self.rollout_set_size = int(self.budget / self.rollouts_per_action / (self.rollout_length+1) / 32)
            print("Given the budget of", self.budget, " the rollout set size will be:", self.rollout_set_size)
        else:
            self.rollout_set_size = rollout_set_size  # number of states sampled from rollout_set D_k
            assert(self.budget == self.rollout_set_size * self.rollouts_per_action * (self.rollout_length+1) * 32)
        self.gamma = gamma
        self.num_features = num_features
        self.num_value_features = num_value_features

    def construct_rollout_set(self):
        self.rollout_set = np.random.choice(a=self.rollout_state_population, size=self.rollout_set_size, replace=False)

    def perform_rollouts(self, policy_weights, value_weights, generative_model):
        self.construct_rollout_set()
        state_features = np.zeros((self.rollout_set_size, self.num_value_features), dtype=np.float)
        state_values = np.zeros(self.rollout_set_size, dtype=np.float)
        state_action_values = np.zeros((self.rollout_set_size, 34), dtype=np.float)
        state_action_features = np.zeros((self.rollout_set_size, 34, self.num_features))
        num_available_actions = np.zeros(self.rollout_set_size, dtype=np.int64)
        did_rollout = np.ones(self.rollout_set_size, dtype=bool)
        for ix, rollout_state in enumerate(self.rollout_set):
            # TODO: implement self.rollouts_per_action...    however, in Scherrer et al. (2015) this always values 1

            # Rollouts for state-value function estimation
            state_features[ix, :] = rollout_state.get_features_pure(True)[1:]  # Don't store intercept
            state_values[ix] = value_roll_out(rollout_state, self.rollout_length, self.gamma, generative_model,
                                              policy_weights, value_weights, self.num_features)

            # Rollouts for action-value function estimation
            actions_value_estimates, state_action_features_ix = \
                action_value_roll_out(rollout_state, self.rollout_length, self.gamma, generative_model,
                                      policy_weights, value_weights, self.num_features)
            num_av_acts = len(actions_value_estimates)
            num_available_actions[ix] = num_av_acts
            state_action_values[ix, :num_av_acts] = actions_value_estimates
            if num_av_acts > 0:
                state_action_features[ix, :num_av_acts, :] = state_action_features_ix
            else:
                did_rollout[ix] = False
        return state_features, state_values, state_action_features, state_action_values, did_rollout, num_available_actions


class Cbmpi:
    def __init__(self,
                 policy_approximator,
                 value_function_approximator,
                 generative_model,
                 rollout_handler,
                 verbose):
        self.name = "cbmpi"
        self.policy_approximator = policy_approximator
        self.value_function_approximator = value_function_approximator  # TODO
        self.generative_model = generative_model
        self.rollout_handler = rollout_handler

        self.num_features = self.policy_approximator.num_features
        self.num_value_features = self.value_function_approximator.num_value_features
        self.policy_weights = policy_approximator.policy_weights
        self.value_weights = value_function_approximator.value_weights

        self.verbose = verbose
        self.is_learning = True
        self.step = 0

    def update_steps(self):
        """ Generic function called from certain RL run procedures. Pretty boring and useless here!"""
        self.step += 1

    def learn(self):
        start_time = time.time()
        state_features, state_values, state_action_features, state_action_values, did_rollout, num_available_actions = \
            self.rollout_handler.perform_rollouts(self.policy_weights, self.value_weights, self.generative_model)
        if self.verbose:
            print("Rollouts took " + str((time.time() - start_time) / 60) + " minutes.")

        start_time = time.time()
        self.value_weights = self.value_function_approximator.fit(state_features, state_values)
        self.policy_weights = self.policy_approximator.fit(state_action_features, state_action_values, did_rollout, num_available_actions)

        if self.verbose:
            print("Function approximation took " + str((time.time() - start_time) / 60) + " minutes.")
            print("New value_weights: ", self.value_weights)
            print("New policy_weights: ", self.policy_weights)


# @njit(cache=False)
def policy_loss_function(pol_weights, N, did_rollout, state_action_features, num_available_actions,
                         state_action_values):
    loss = 0.
    number_of_samples = 0
    for state_ix in range(N):
        if did_rollout[state_ix]:
            values = state_action_features[state_ix, :num_available_actions[state_ix]].dot(pol_weights)
            pol_value = state_action_values[state_ix, np.argmax(values)]
            max_value = np.max(state_action_values[state_ix, :len(values)])
            loss += max_value - pol_value
            number_of_samples += 1
    loss /= number_of_samples
    return loss


# @njit(cache=False)
def action_value_roll_out(start_state,
                          m,
                          gamma,
                          generative_model,
                          policy_weights,
                          value_weights,
                          num_features):
    generative_model.next_tetromino()
    child_states = generative_model.get_after_states(start_state)
    num_child_states = len(child_states)
    action_value_estimates = np.zeros(num_child_states)
    state_action_features = np.zeros((num_child_states, num_features))
    if num_child_states == 0:
        # TODO: check whether this (returning zeros) has any side effects on learning...
        return action_value_estimates, state_action_features
    for child_ix in range(num_child_states):
        state_tmp = child_states[child_ix]
        state_action_features[child_ix] = state_tmp.get_features_pure(False) # order_by=None, standardize_by=None
        cumulative_reward = state_tmp.n_cleared_lines
        # print("Starting new action rollout")
        game_ended = False
        count = 0
        while not game_ended and count < m:  # there are m rollouts
            generative_model.next_tetromino()
            available_after_states = generative_model.get_after_states(state_tmp)
            num_after_states = len(available_after_states)
            if num_after_states == 0:
                game_ended = True
            else:
                state_tmp = choose_action_in_rollout(available_after_states, policy_weights, num_features)
                cumulative_reward += (gamma ** count) * state_tmp.n_cleared_lines
            count += 1

        # One more (the (m+1)-th) for truncation value!
        if not game_ended:
            generative_model.next_tetromino()
            available_after_states = generative_model.get_after_states(state_tmp)
            num_after_states = len(available_after_states)
            if num_after_states > 0:
                state_tmp = choose_action_in_rollout(available_after_states, policy_weights, num_features)
                final_state_features = state_tmp.get_features_pure(True)
                cumulative_reward += (gamma ** count) * final_state_features.dot(value_weights)

        action_value_estimates[child_ix] = cumulative_reward
    return action_value_estimates, state_action_features


# @njit(cache=False)
def value_roll_out(start_state,
                   m,
                   gamma,
                   generative_model,
                   policy_weights,
                   value_weights,
                   num_features):
    value_estimate = 0.0
    state_tmp = start_state
    count = 0
    while not state_tmp.terminal_state and count < m:  # there are only (m-1) rollouts
        generative_model.next_tetromino()
        available_after_states = generative_model.get_after_states(state_tmp)
        if len(available_after_states) == 0:
            return value_estimate
        state_tmp = choose_action_in_rollout(available_after_states, policy_weights,
                                             # rollout_dom_filter, rollout_cumu_dom_filter,
                                             num_features)
        value_estimate += gamma ** count * state_tmp.n_cleared_lines
        count += 1

    # One more (the m-th) for truncation value!
    if not state_tmp.terminal_state:
        generative_model.next_tetromino()
        available_after_states = generative_model.get_after_states(state_tmp)
        if len(available_after_states) == 0:
            return value_estimate
        state_tmp = choose_action_in_rollout(available_after_states, policy_weights,
                                             # rollout_dom_filter, rollout_cumu_dom_filter,
                                             num_features)
        final_state_features = state_tmp.get_features_pure(True)  # order_by=None, standardize_by=None,
        # value_estimate += self.gamma ** count + final_state_features.dot(self.value_weights)
        value_estimate += (gamma ** count) * final_state_features.dot(value_weights)
    return value_estimate


# @njit(cache=False)
def choose_action_in_rollout(available_after_states, policy_weights,
                             # rollout_dom_filter, rollout_cumu_dom_filter,
                             # feature_directors,
                             num_features):
    num_states = len(available_after_states)
    action_features = np.zeros((num_states, num_features))
    for ix, after_state in enumerate(available_after_states):
        action_features[ix] = after_state.get_features_pure(False)  # , order_by=self.feature_order
    # if rollout_cumu_dom_filter:
    #     not_simply_dominated, not_cumu_dominated = dom_filter(action_features, len_after_states=num_states)  # domtools.
    #     action_features = action_features[not_cumu_dominated]
    #     map_back_vector = np.nonzero(not_cumu_dominated)[0]
    #     # if rollout_cumu_dom_filter:
    #     #     available_after_states = available_after_states[not_simply_dominated]
    #     #     action_features = action_features[not_simply_dominated]
    #     # elif rollout_dom_filter:
    #     #     available_after_states = available_after_states[not_cumu_dominated]
    #     #     action_features = action_features[not_cumu_dominated]
    # else:
    #     raise ValueError("Currently only implemented with cumu_dom_filter")
    utilities = action_features.dot(np.ascontiguousarray(policy_weights))
    move_index = np.argmax(utilities)
    move = available_after_states[move_index]
    return move