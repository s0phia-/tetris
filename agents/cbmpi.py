import numpy as np
from stew import StewMultinomialLogit, ChoiceSetData
from tetris import tetromino
from numba import njit
from sklearn.linear_model import LinearRegression
import gc
import cma
import time


class Cbmpi:
    def __init__(self, m, N, M, B, D,
                 feature_type, num_columns,
                 cmaes_var, min_iterations,
                 verbose, seed=0, id="",
                 discrete_choice=False):
        self.name = "cbmpi"
        self.is_learning = True

        self.num_features = 8
        self.num_value_features = 13
        self.num_columns = num_columns
        self.feature_type = feature_type
        self.verbose = verbose
        self.max_choice_set_size = 34
        self.tetromino_handler = tetromino.Tetromino(self.feature_type, self.num_features, self.num_columns)

        self.m = m  # rollout_size
        self.M = M  # number of independent rollouts per
        assert(self.M == 1)
        self.B = B  # budget
        if N is None:
            self.N = int(self.B / self.M / (self.m+1) / 32)
            print("Given the budget of", self.B, " the rollout set size will be:", self.N)
        else:
            self.N = N  # number of states sampled from rollout_set D_k
        self.D = D  # rollout set
        self.D_k = None

        # print("The budget is", self.B)
        self.gamma = 1
        self.eta = 0  # CMA-ES parameter (dunno where to set...)
        self.zeta = 0.5  # CMA-ES parameter ( // 2 is standard in cma)
        self.n = int(self.num_features * 15)  # CMA-ES parameter ('popsize' in cma)
        self.cmaes_var = cmaes_var
        self.min_iterations = min_iterations

        self.policy_weights = np.random.normal(loc=0, scale=self.cmaes_var, size=self.num_features)
        self.value_weights = np.zeros(self.num_value_features + 1)

        self.lin_reg = LinearRegression(n_jobs=1, fit_intercept=True)  # n_jobs must be 1... otherwise clashes with multiprocessing.Pool
        self.lin_reg.coef_ = np.zeros(self.num_value_features)
        self.lin_reg.intercept_ = 0.

        self.seed = seed

        self.discrete_choice = discrete_choice
        if self.discrete_choice:
            self.regularization = "no_regularization"
            self.model = StewMultinomialLogit(num_features=self.num_features)
            self.mlogit_data = ChoiceSetData(num_features=self.num_features, max_choice_set_size=self.max_choice_set_size)
        else:
            self.cma_es = cma.CMAEvolutionStrategy(
                np.random.normal(loc=0, scale=1, size=self.num_features),
                self.cmaes_var, inopts={'verb_disp': 0,
                                        'verb_filenameprefix': "output/cmaesout" + str(self.seed),
                                        'popsize': self.n})
        self.step = 0

    def update_steps(self):
        self.step += 1

    def learn(self):
        self.construct_rollout_set()
        if self.verbose:
            print("Rollout set constructed")
        start_time = time.time()
        self.state_features = np.zeros((self.N, self.num_value_features), dtype=np.float)
        self.state_values = np.zeros(self.N, dtype=np.float)
        self.state_action_values = np.zeros((self.N, 34), dtype=np.float)
        # self.state_action_features = []
        self.state_action_features = np.zeros((self.N, 34, self.num_features))
        self.num_available_actions = np.zeros(self.N, dtype=np.int64)
        self.did_rollout = np.ones(self.N, dtype=bool)
        for ix, rollout_state in enumerate(self.D_k):
            # if ix % 5000 == 0 and self.verbose:
            #     print("rollout state:", ix)
            # Don't store intercept
            self.state_features[ix, :] = rollout_state.get_features_pure(True)[1:]
            self.state_values[ix] = value_roll_out(rollout_state, self.m, self.gamma, self.tetromino_handler,
                                                   self.policy_weights, self.value_weights, self.num_features)
            actions_value_estimates, state_action_features = \
                action_value_roll_out(rollout_state, self.m, self.gamma, self.tetromino_handler,
                                      self.policy_weights, self.value_weights, self.num_features)
            num_available_actions = len(actions_value_estimates)
            self.num_available_actions[ix] = num_available_actions
            self.state_action_values[ix, :num_available_actions] = actions_value_estimates
            if num_available_actions > 0:
                self.state_action_features[ix, :num_available_actions, :] = state_action_features
            else:
                print("DID NOT DO ROLLOUT")
                self.did_rollout[ix] = False
            # self.state_action_features.append(state_action_features)

        self.delete_rollout_set()
        end_time = time.time()
        if self.verbose:
            print("Rollouts took " + str((end_time - start_time) / 60) + " minutes.")
        assert(len(self.state_action_features) == self.N)
        # # Max
        # self.max_state_action_values = self.state_action_values.max(axis=1)

        if self.verbose:
            print("Estimating weights now...")

        # Approximate value function
        if self.verbose:
            print("Number of self.state_values", len(self.state_values))
        self.lin_reg.fit(self.state_features, self.state_values)
        self.value_weights = np.hstack((self.lin_reg.intercept_, self.lin_reg.coef_))
        if self.verbose:
            print("new value weights: ", self.value_weights)

        # Approximate policy
        start_time = time.time()
        if self.discrete_choice:
            self.mlogit_data.delete_data()
            # stacked_features = np.concatenate([self.state_action_features, axis=0)
            # TODO: rewrite to use less copying / memory...
            for state_ix in range(self.N):
                state_act_feat_ix = self.state_action_features[state_ix]
                if state_act_feat_ix is not None:
                    action_values = self.state_action_values[state_ix, :len(state_act_feat_ix)]
                    self.mlogit_data.push(features=self.state_action_features[state_ix], choice_index=np.argmax(action_values), delete_oldest=False)
            if self.ols:
                self.policy_weights = self.model.fit(data=self.mlogit_data.sample(), lam=0, standardize=False)
            # else:
            #     self.policy_weights, _ = self.model.cv_fit(data=self.mlogit_data.sample(), standardize=self.standardize_features)
        else:
            self.cma_es = cma.CMAEvolutionStrategy(np.random.normal(loc=0, scale=1, size=self.num_features), self.cmaes_var,
                                                   inopts={'verb_disp': 0,
                                                           'verb_filenameprefix': "output/cmaesout" + str(self.seed),
                                                           'popsize': self.n})
            # self.policy_weights = self.cma_es.optimize(self.policy_loss_function, min_iterations=1).result.xbest
            self.policy_weights = self.cma_es.optimize(lambda x: policy_loss_function(x, self.N, self.did_rollout, self.state_action_features,
                                                                                      self.num_available_actions, self.state_action_values),
                                                       min_iterations=self.min_iterations).result.xbest
        end_time = time.time()
        if self.verbose:
            print("CMAES took " + str((end_time - start_time) / 60) + " minutes.")
            print("new policy_weights: ", self.policy_weights)

    def construct_rollout_set(self):
        self.D_k = np.random.choice(a=self.D, size=self.N, replace=False)

    def delete_rollout_set(self):
        self.D_k = None
        gc.collect()

    def policy_loss_function(self, pol_weights):
        loss = 0.
        number_of_samples = 0
        for state_ix in range(self.N):
            # print("self.N", self.N)
            if self.did_rollout[state_ix]:
                values = self.state_action_features[state_ix, :self.num_available_actions[state_ix]].dot(pol_weights)
                # print("values", values)
                pol_value = self.state_action_values[state_ix, np.argmax(values)]
                max_value = np.max(self.state_action_values[state_ix, :len(values)])
                # print("state_ix", state_ix)
                # print("self.state_action_features[state_ix]", self.state_action_features[state_ix])
                # print("pol_weights", pol_weights)
                loss += max_value - pol_value
                number_of_samples += 1
            # else:
            #     pass
            #     # print(state_ix, " has no action features / did not produce a rollout!")
        loss /= number_of_samples
        # print("loss", loss)
        return loss


@njit(cache=False)
def policy_loss_function(pol_weights, N, did_rollout, state_action_features, num_available_actions,
                         state_action_values):
    loss = 0.
    number_of_samples = 0
    for state_ix in range(N):
        # print("self.N", self.N)
        if did_rollout[state_ix]:
            values = state_action_features[state_ix, :num_available_actions[state_ix]].dot(pol_weights)
            # print("values", values)
            pol_value = state_action_values[state_ix, np.argmax(values)]
            max_value = np.max(state_action_values[state_ix, :len(values)])
            # print("state_ix", state_ix)
            # print("self.state_action_features[state_ix]", self.state_action_features[state_ix])
            # print("pol_weights", pol_weights)
            loss += max_value - pol_value
            number_of_samples += 1
        # else:
        #     pass
        #     # print(state_ix, " has no action features / did not produce a rollout!")
    loss /= number_of_samples
    # print("loss", loss)
    return loss


@njit(cache=False)
def action_value_roll_out(start_state,
                          m,
                          gamma,
                          tetromino_handler,
                          policy_weights,
                          value_weights,
                          num_features):
    tetromino_handler.next_tetromino()
    child_states = tetromino_handler.get_after_states(start_state)
    num_child_states = len(child_states)
    action_value_estimates = np.zeros(num_child_states)
    state_action_features = np.zeros((num_child_states, num_features))
    if num_child_states == 0:
        # TODO: check whether this (returning zeros) has any side effects on learning...
        print("")
        return action_value_estimates, state_action_features
    for child_ix in range(num_child_states):
        state_tmp = child_states[child_ix]
        state_action_features[child_ix] = state_tmp.get_features_pure(False) # order_by=None, standardize_by=None
        cumulative_reward = state_tmp.n_cleared_lines
        # print("Starting new action rollout")
        game_ended = False
        count = 0
        while not game_ended and count < m:  # there are m rollouts
            tetromino_handler.next_tetromino()
            available_after_states = tetromino_handler.get_after_states(state_tmp)
            num_after_states = len(available_after_states)
            if num_after_states == 0:
                game_ended = True
            else:
                state_tmp = choose_action_in_rollout(available_after_states, policy_weights, num_features)
                cumulative_reward += (gamma ** count) * state_tmp.n_cleared_lines
            count += 1

        # One more (the (m+1)-th) for truncation value!
        if not game_ended:
            tetromino_handler.next_tetromino()
            available_after_states = tetromino_handler.get_after_states(state_tmp)
            num_after_states = len(available_after_states)
            if num_after_states > 0:
                state_tmp = choose_action_in_rollout(available_after_states, policy_weights, num_features)
                final_state_features = state_tmp.get_features_pure(True)
                cumulative_reward += (gamma ** count) * final_state_features.dot(value_weights)

        action_value_estimates[child_ix] = cumulative_reward
    return action_value_estimates, state_action_features


@njit(cache=False)
def value_roll_out(start_state,
                   m,
                   gamma,
                   tetromino_handler,
                   policy_weights,
                   value_weights,
                   num_features):
    value_estimate = 0.0
    state_tmp = start_state
    count = 0
    while not state_tmp.terminal_state and count < m:  # there are only (m-1) rollouts
        tetromino_handler.next_tetromino()
        available_after_states = tetromino_handler.get_after_states(state_tmp)
        if len(available_after_states) == 0:
            return value_estimate
        state_tmp = choose_action_in_rollout(available_after_states, policy_weights,
                                             # rollout_dom_filter, rollout_cumu_dom_filter,
                                             num_features)
        value_estimate += gamma ** count * state_tmp.n_cleared_lines
        count += 1

    # One more (the m-th) for truncation value!
    if not state_tmp.terminal_state:
        tetromino_handler.next_tetromino()
        available_after_states = tetromino_handler.get_after_states(state_tmp)
        if len(available_after_states) == 0:
            return value_estimate
        state_tmp = choose_action_in_rollout(available_after_states, policy_weights,
                                             # rollout_dom_filter, rollout_cumu_dom_filter,
                                             num_features)
        final_state_features = state_tmp.get_features_pure(True)  # order_by=None, standardize_by=None,
        # value_estimate += self.gamma ** count + final_state_features.dot(self.value_weights)
        value_estimate += (gamma ** count) * final_state_features.dot(value_weights)
    return value_estimate


@njit(cache=False)
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