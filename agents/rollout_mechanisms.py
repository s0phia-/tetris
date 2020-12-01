import numpy as np
from domtools import dom_filter as dominance_filter
from numba import njit
import warnings


class OnlineRollout:
    def __init__(self,
                 rollout_length,
                 rollouts_per_action,
                 num_features,
                 num_value_features,
                 rollout_set_size=None,
                 gamma=1):
        self.name = "OnlineRollout"
        self.rollout_length = rollout_length
        self.rollouts_per_action = rollouts_per_action
        self.rollout_set_size = rollout_set_size
        self.budget = self.rollout_length * self.rollouts_per_action * self.rollout_set_size * 32
        self.gamma = gamma


class BatchRollout:
    def __init__(self,
                 rollout_state_population,
                 rollout_length,
                 rollouts_per_action,
                 rollout_set_size,
                 num_features,
                 num_value_features,
                 reward_greedy,
                 feature_directors,
                 use_dom=False,
                 use_cumul_dom=False,
                 use_filters_during_rollout=False,
                 use_filters_before_rollout=False,
                 gamma=0.9):
        self.name = "BatchRollout"
        self.rollout_state_population = rollout_state_population
        self.rollout_set = None  # use self.construct_rollout_set()
        self.rollout_length = rollout_length
        self.rollouts_per_action = rollouts_per_action
        self.rollout_set_size = rollout_set_size
        self.gamma = gamma
        self.num_features = num_features
        self.num_value_features = num_value_features
        self.reward_greedy = reward_greedy
        assert not self.reward_greedy, "Reward_greedy currently not supported (see rollout functions)."

        """
        About dominance filtering: it is currently possible to filter before rollouts are started, during
        rollouts, or both. In any case all filters are either "simple dominance" filters or "cumulative
        dominance" filters. They cannot be mixed (e.g., simple for before and cumulative during rollouts).  
        """
        self.use_dom = use_dom
        self.use_cumul_dom = use_cumul_dom
        assert not (self.use_dom and self.use_cumul_dom)
        self.use_filters_during_rollout = use_filters_during_rollout
        self.use_filters_before_rollout = use_filters_before_rollout

        # Feature directors are only used for filtering.
        self.feature_directors = feature_directors

    def construct_rollout_set(self):
        self.rollout_set = np.random.choice(a=self.rollout_state_population, size=self.rollout_set_size,
                                            replace=False if len(self.rollout_state_population) > self.rollout_set_size else True)

    def perform_rollouts(self, policy_weights, value_weights, generative_model, use_state_values):
        self.construct_rollout_set()
        state_features = np.zeros((self.rollout_set_size, self.num_value_features), dtype=np.float)
        state_values = np.zeros(self.rollout_set_size, dtype=np.float)
        state_action_values = np.zeros((self.rollout_set_size, 34), dtype=np.float)
        state_action_features = np.zeros((self.rollout_set_size, 34, self.num_features))
        num_available_actions = np.zeros(self.rollout_set_size, dtype=np.int64)
        did_rollout = np.ones(self.rollout_set_size, dtype=bool)
        for ix, rollout_state in enumerate(self.rollout_set):
            # print(f"rollout state = {ix}")
            # Sample tetromino for each rollout state (same for state and state-action rollouts)
            generative_model.next_tetromino()

            if use_state_values:
                # Rollouts for state-value function estimation
                state_features[ix, :] = rollout_state.get_features_pure(True)[1:]  # Don't store intercept
                state_values[ix] = value_roll_out(rollout_state, self.rollout_length, self.gamma,
                                                  generative_model.copy_with_same_current_tetromino(),
                                                  policy_weights, value_weights, self.num_features,
                                                  self.reward_greedy, self.use_filters_during_rollout,
                                                  self.use_dom, self.use_cumul_dom,
                                                  self.feature_directors)

            # Rollouts for action-value function estimation
            actions_value_estimates, state_action_features_ix = \
                general_action_value_rollout(self.use_filters_during_rollout,
                                             self.use_filters_before_rollout,
                                             rollout_state,
                                             self.rollout_length,
                                             self.rollouts_per_action,
                                             self.gamma,
                                             generative_model.copy_with_same_current_tetromino(),
                                             policy_weights,
                                             value_weights,
                                             self.num_features,
                                             use_state_values,
                                             self.reward_greedy,
                                             self.use_dom,
                                             self.use_cumul_dom,
                                             self.feature_directors)
            # if (self.use_dom or self.use_cumul_dom) and not self.use_filters_during_rollout:
            #     # Use dominance filters to filter the actions-to-be-considered, i.e., the initial A(s).
            #     actions_value_estimates, state_action_features_ix = \
            #         action_value_roll_out_with_filters(rollout_state, self.rollout_length, self.rollouts_per_action, self.gamma,
            #                                            generative_model.copy_with_same_current_tetromino(),
            #                                            policy_weights, value_weights, self.num_features, use_state_values,
            #                                            self.reward_greedy, self.use_dom, self.use_cumul_dom, self.feature_directors)
            # elif self.use_filters_during_rollout:
            #     # Evaluate all actions in A(s), but use rollouts during filters.
            #     actions_value_estimates, state_action_features_ix = \
            #         action_value_roll_out_with_filters_during_rollouts(rollout_state, self.rollout_length, self.rollouts_per_action, self.gamma,
            #                                                            generative_model.copy_with_same_current_tetromino(),
            #                                                            policy_weights, value_weights, self.num_features, use_state_values,
            #                                                            self.reward_greedy, self.use_dom, self.use_cumul_dom, self.feature_directors)
            # else:
            #     actions_value_estimates, state_action_features_ix = \
            #         action_value_roll_out(rollout_state, self.rollout_length, self.rollouts_per_action, self.gamma,
            #                               generative_model.copy_with_same_current_tetromino(),
            #                               policy_weights, value_weights, self.num_features, use_state_values,
            #                               self.reward_greedy)
            num_av_acts = len(actions_value_estimates)
            num_available_actions[ix] = num_av_acts
            state_action_values[ix, :num_av_acts] = actions_value_estimates
            if num_av_acts > 0:
                state_action_features[ix, :num_av_acts, :] = state_action_features_ix
            else:
                # Rollout starting state was terminal state.
                did_rollout[ix] = False
        return dict(state_features=state_features,
                    state_values=state_values,
                    state_action_features=state_action_features,
                    state_action_values=state_action_values,
                    did_rollout=did_rollout,
                    num_available_actions=num_available_actions)


@njit(cache=False)
def general_action_value_rollout(use_filters_during_rollout,
                                 use_filters_before_rollout,
                                 start_state,
                                 rollout_length,
                                 rollouts_per_action,
                                 gamma,
                                 generative_model,
                                 policy_weights,
                                 value_weights,
                                 num_features,
                                 use_state_values,
                                 reward_greedy,
                                 use_dom,
                                 use_cumul_dom,
                                 feature_directors):
    child_states = generative_model.get_after_states(start_state)
    num_child_states = len(child_states)
    action_value_estimates = np.zeros(num_child_states)
    state_action_features = np.zeros((num_child_states, num_features))

    if num_child_states == 0:
        # Rollout starting state is terminal state
        return action_value_estimates, state_action_features

    state_action_features = np.zeros((num_child_states, num_features), dtype=np.float_)
    for ix in range(num_child_states):
        state_action_features[ix] = child_states[ix].get_features_pure(False)  # , order_by=self.feature_order

    if use_filters_before_rollout:
        not_simply_dominated, not_cumu_dominated = dominance_filter(state_action_features * feature_directors,
                                                                    len_after_states=num_child_states)

    is_not_filtered_out = np.ones(num_child_states, dtype=np.bool_)
    for child_ix in range(num_child_states):
        do_rollout = False
        if use_filters_before_rollout:
            if use_dom:
                if not_simply_dominated[child_ix]:
                    do_rollout = True
            elif use_cumul_dom:
                if not_cumu_dominated[child_ix]:
                    do_rollout = True
            else:
                raise ValueError
        else:
            do_rollout = True

        if do_rollout:
            state_tmp = child_states[child_ix]
            start_reward = state_tmp.n_cleared_lines

            for rollout_ix in range(rollouts_per_action):
                cumulative_reward = start_reward
                game_ended = False
                count = 0
                while not game_ended and count < rollout_length:  # there are rollout_length rollouts
                    generative_model.next_tetromino()
                    available_after_states = generative_model.get_after_states(state_tmp)
                    num_after_states = len(available_after_states)
                    if num_after_states == 0:
                        # Terminal state
                        game_ended = True
                    else:
                        state_tmp = select_action_in_rollout(available_after_states, policy_weights,
                                                             num_features, use_filters_during_rollout, feature_directors,
                                                             use_dom, use_cumul_dom)
                        cumulative_reward += (gamma ** count) * state_tmp.n_cleared_lines
                    count += 1

                # One more (the (rollout_length+1)-th) for truncation value!
                if use_state_values and not game_ended:
                    generative_model.next_tetromino()
                    available_after_states = generative_model.get_after_states(state_tmp)
                    num_after_states = len(available_after_states)
                    if num_after_states > 0:
                        state_tmp = select_action_in_rollout(available_after_states, policy_weights,
                                                             num_features, use_filters_during_rollout, feature_directors,
                                                             use_dom, use_cumul_dom)

                        # Get state value of last state.
                        final_state_features = state_tmp.get_features_pure(True)
                        cumulative_reward += (gamma ** count) * final_state_features.dot(value_weights)

                action_value_estimates[child_ix] += cumulative_reward
        else:
            is_not_filtered_out[child_ix] = False
            action_value_estimates[child_ix] = -np.inf

    action_value_estimates = action_value_estimates[is_not_filtered_out]
    state_action_features = state_action_features[is_not_filtered_out]
    action_value_estimates /= rollouts_per_action
    return action_value_estimates, state_action_features


@njit
def select_action_in_rollout(available_after_states, policy_weights, num_features,
                             use_filters_during_rollout, feature_directors, use_dom, use_cumul_dom):
    num_after_states = len(available_after_states)
    action_features = np.zeros((num_after_states, num_features))
    for ix, after_state in enumerate(available_after_states):
        action_features[ix] = after_state.get_features_pure(False)  # , order_by=self.feature_order
    if use_filters_during_rollout:
        not_simply_dominated, not_cumu_dominated = dominance_filter(action_features * feature_directors,
                                                                    len_after_states=num_after_states)  # domtools.
        if use_dom:
            action_features = action_features[not_simply_dominated]
            map_back_vector = np.nonzero(not_simply_dominated)[0]
        elif use_cumul_dom:
            action_features = action_features[not_cumu_dominated]
            map_back_vector = np.nonzero(not_cumu_dominated)[0]
        else:
            raise ValueError("Either use_dom or use_cumul_dom has to be true.")

    utilities = action_features.dot(np.ascontiguousarray(policy_weights))
    move_index = np.argmax(utilities)
    if use_filters_during_rollout:
        state_tmp = available_after_states[map_back_vector[move_index]]
    else:
        state_tmp = available_after_states[move_index]
    return state_tmp


@njit
def value_roll_out(start_state,
                   rollout_length,
                   gamma,
                   generative_model,
                   policy_weights,
                   value_weights,
                   num_features,
                   reward_greedy,
                   use_filters_during_rollout,
                   use_dom,
                   use_cumul_dom,
                   feature_directors):
    value_estimate = 0.0
    state_tmp = start_state
    count = 0
    while not state_tmp.terminal_state and count < rollout_length:  # there are only (m-1) rollouts
        # generative_model.next_tetromino()
        available_after_states = generative_model.get_after_states(state_tmp)
        num_after_states = len(available_after_states)
        if num_after_states == 0:
            return value_estimate
        state_tmp = select_action_in_rollout(available_after_states, policy_weights, num_features,
                                             use_filters_during_rollout, feature_directors, use_dom, use_cumul_dom)
        value_estimate += (gamma ** count) * state_tmp.n_cleared_lines
        count += 1
        generative_model.next_tetromino()

    # One more (the m-th) for truncation value!
    if not state_tmp.terminal_state:
        # generative_model.next_tetromino()
        available_after_states = generative_model.get_after_states(state_tmp)
        num_after_states = len(available_after_states)
        if num_after_states == 0:
            return value_estimate
        state_tmp = select_action_in_rollout(available_after_states, policy_weights, num_features,
                                             use_filters_during_rollout, feature_directors, use_dom, use_cumul_dom)
        final_state_features = state_tmp.get_features_pure(True)  # order_by=None, standardize_by=None,
        value_estimate += (gamma ** count) * final_state_features.dot(value_weights)
    return value_estimate


def calculate_available_actions(rollout_state_population, generative_model, env):
    print("Calculate available actions with and without filters")
    num_av_acts = np.zeros(len(rollout_state_population))
    num_fil_av_acts = np.zeros(len(rollout_state_population))
    feature_directors = np.array([-1, -1, -1, -1, -1, -1, 1, -1], dtype=np.float64)
    for ix in range(len(rollout_state_population)):
        # print(ix)
        generative_model.next_tetromino()
        child_states = generative_model.get_after_states(rollout_state_population[ix])
        num_child_states = len(child_states)
        num_av_acts[ix] = len(child_states)

        state_action_features = np.zeros((num_child_states, env.num_features), dtype=np.float_)
        for child_ix in range(num_child_states):
            state_action_features[child_ix] = child_states[child_ix].get_features_pure(False)  # , order_by=self.feature_order

        not_simply_dominated, not_cumu_dominated = dominance_filter(state_action_features * feature_directors,
                                                                    len_after_states=num_child_states)
        num_fil_av_acts[ix] = np.sum(not_cumu_dominated)

    print(f"The mean number of available actions was {np.mean(num_av_acts)}")
    print(f"The mean number of FILTERED available actions was {np.mean(num_fil_av_acts)}")


# OLD SPLIT-UP rollout functions

# @njit(cache=False)
# def action_value_roll_out_with_filters_during_rollouts(start_state,
#                                                        rollout_length,
#                                                        rollouts_per_action,
#                                                        gamma,
#                                                        generative_model,
#                                                        policy_weights,
#                                                        value_weights,
#                                                        num_features,
#                                                        use_state_values,
#                                                        reward_greedy,
#                                                        use_dom,
#                                                        use_cumul_dom,
#                                                        feature_directors):
#
#     if reward_greedy:
#         raise ValueError("Reward-greedy not implemented for rollouts with filters during rollouts.")
#
#     child_states = generative_model.get_after_states(start_state)
#     num_child_states = len(child_states)
#     action_value_estimates = np.zeros(num_child_states)
#     state_action_features = np.zeros((num_child_states, num_features))
#
#     if num_child_states == 0:
#         return action_value_estimates, state_action_features
#     for child_ix in range(num_child_states):
#         state_tmp = child_states[child_ix]
#         start_reward = state_tmp.n_cleared_lines
#         state_action_features[child_ix] = state_tmp.get_features_pure(False)  # order_by=None, standardize_by=None
#         for rollout_ix in range(rollouts_per_action):
#             cumulative_reward = start_reward
#             game_ended = False
#             count = 0
#             while not game_ended and count < rollout_length:  # there are rollout_length rollouts
#                 generative_model.next_tetromino()
#                 available_after_states = generative_model.get_after_states(state_tmp)
#                 num_after_states = len(available_after_states)
#                 if num_after_states == 0:
#                     game_ended = True
#                 else:
#                     state_tmp = choose_action_in_rollout_with_filters(available_after_states, policy_weights,
#                                                                       num_features, reward_greedy,
#                                                                       use_dom, use_cumul_dom,
#                                                                       feature_directors)
#                     cumulative_reward += (gamma ** count) * state_tmp.n_cleared_lines
#                 count += 1
#
#             # One more (the (rollout_length+1)-th) for truncation value!
#             if use_state_values and not game_ended:
#                 generative_model.next_tetromino()
#                 available_after_states = generative_model.get_after_states(state_tmp)
#                 num_after_states = len(available_after_states)
#                 if num_after_states > 0:
#                     state_tmp = choose_action_in_rollout_with_filters(available_after_states, policy_weights,
#                                                                       num_features, reward_greedy,
#                                                                       use_dom, use_cumul_dom,
#                                                                       feature_directors)
#                     final_state_features = state_tmp.get_features_pure(True)
#                     cumulative_reward += (gamma ** count) * final_state_features.dot(value_weights)
#
#             action_value_estimates[child_ix] += cumulative_reward
#     action_value_estimates /= rollouts_per_action
#     return action_value_estimates, state_action_features
#
#
# @njit(cache=False)
# def action_value_roll_out_with_filters(start_state,
#                                        rollout_length,
#                                        rollouts_per_action,
#                                        gamma,
#                                        generative_model,
#                                        policy_weights,
#                                        value_weights,
#                                        num_features,
#                                        use_state_values,
#                                        reward_greedy,
#                                        use_dom,
#                                        use_cumul_dom,
#                                        feature_directors):
#     # generative_model.next_tetromino()
#     child_states = generative_model.get_after_states(start_state)
#     num_child_states = len(child_states)
#     action_value_estimates = np.zeros(num_child_states)
#     state_action_features = np.zeros((num_child_states, num_features))
#
#     if num_child_states == 0:
#         # TODO: check whether this (returning zeros) has any side effects on learning...
#         return action_value_estimates, state_action_features
#
#     state_action_features = np.zeros((num_child_states, num_features), dtype=np.float_)
#     for ix in range(num_child_states):
#         state_action_features[ix] = child_states[ix].get_features_pure(False)  # , order_by=self.feature_order
#
#     not_simply_dominated, not_cumu_dominated = dominance_filter(state_action_features * feature_directors, len_after_states=num_child_states)
#
#     is_not_filtered_out = np.ones(num_child_states, dtype=np.bool_)
#     for child_ix in range(num_child_states):
#         do_rollout = False
#         if use_cumul_dom:
#             if not_cumu_dominated[child_ix]:
#                 do_rollout = True
#         elif use_dom:
#             if not_simply_dominated[child_ix]:
#                 do_rollout = True
#         else:
#             do_rollout = True
#
#         if do_rollout:
#             state_tmp = child_states[child_ix]
#             start_reward = state_tmp.n_cleared_lines
#             # state_action_features[child_ix] = state_tmp.get_features_pure(False)  # order_by=None, standardize_by=None
#
#             for rollout_ix in range(rollouts_per_action):
#                 cumulative_reward = start_reward
#                 # print("Starting new action rollout")
#                 game_ended = False
#                 count = 0
#                 while not game_ended and count < rollout_length:  # there are rollout_length rollouts
#                     generative_model.next_tetromino()
#                     available_after_states = generative_model.get_after_states(state_tmp)
#                     num_after_states = len(available_after_states)
#                     if num_after_states == 0:
#                         game_ended = True
#                     else:
#                         state_tmp = choose_action_in_rollout(available_after_states, policy_weights, num_features, reward_greedy)
#                         cumulative_reward += (gamma ** count) * state_tmp.n_cleared_lines
#                     count += 1
#
#                 # One more (the (rollout_length+1)-th) for truncation value!
#                 if use_state_values and not game_ended:
#                     generative_model.next_tetromino()
#                     available_after_states = generative_model.get_after_states(state_tmp)
#                     num_after_states = len(available_after_states)
#                     if num_after_states > 0:
#                         state_tmp = choose_action_in_rollout(available_after_states, policy_weights, num_features, reward_greedy)
#                         final_state_features = state_tmp.get_features_pure(True)
#                         cumulative_reward += (gamma ** count) * final_state_features.dot(value_weights)
#
#                 action_value_estimates[child_ix] += cumulative_reward
#         else:
#             is_not_filtered_out[child_ix] = False
#             action_value_estimates[child_ix] = -np.inf
#
#     action_value_estimates = action_value_estimates[is_not_filtered_out]
#     state_action_features = state_action_features[is_not_filtered_out]
#     action_value_estimates /= rollouts_per_action
#     return action_value_estimates, state_action_features
#
#
# @njit(cache=False)
# def action_value_roll_out(start_state,
#                           rollout_length,
#                           rollouts_per_action,
#                           gamma,
#                           generative_model,
#                           policy_weights,
#                           value_weights,
#                           num_features,
#                           use_state_values,
#                           reward_greedy):
#     # generative_model.next_tetromino()
#     child_states = generative_model.get_after_states(start_state)
#     num_child_states = len(child_states)
#     action_value_estimates = np.zeros(num_child_states)
#     state_action_features = np.zeros((num_child_states, num_features))
#
#     if num_child_states == 0:
#         # TODO: check whether this (returning zeros) has any side effects on learning...
#         return action_value_estimates, state_action_features
#     for child_ix in range(num_child_states):
#         state_tmp = child_states[child_ix]
#         start_reward = state_tmp.n_cleared_lines
#         state_action_features[child_ix] = state_tmp.get_features_pure(False)  # order_by=None, standardize_by=None
#         for rollout_ix in range(rollouts_per_action):
#             cumulative_reward = start_reward
#             # print("Starting new action rollout")
#             game_ended = False
#             count = 0
#             while not game_ended and count < rollout_length:  # there are rollout_length rollouts
#                 generative_model.next_tetromino()
#                 available_after_states = generative_model.get_after_states(state_tmp)
#                 num_after_states = len(available_after_states)
#                 if num_after_states == 0:
#                     game_ended = True
#                 else:
#                     state_tmp = choose_action_in_rollout(available_after_states, policy_weights, num_features, reward_greedy)
#                     cumulative_reward += (gamma ** count) * state_tmp.n_cleared_lines
#                 count += 1
#
#             # One more (the (rollout_length+1)-th) for truncation value!
#             if use_state_values and not game_ended:
#                 generative_model.next_tetromino()
#                 available_after_states = generative_model.get_after_states(state_tmp)
#                 num_after_states = len(available_after_states)
#                 if num_after_states > 0:
#                     state_tmp = choose_action_in_rollout(available_after_states, policy_weights, num_features, reward_greedy)
#                     final_state_features = state_tmp.get_features_pure(True)
#                     cumulative_reward += (gamma ** count) * final_state_features.dot(value_weights)
#
#             action_value_estimates[child_ix] += cumulative_reward
#     action_value_estimates /= rollouts_per_action
#     return action_value_estimates, state_action_features

# @njit(cache=False)
# def filter_actions(available_after_states, action_features, feature_directors, dom_filter, cumul_dom_filter):
#     not_simply_dominated, not_cumu_dominated = dominance_filter(action_features * feature_directors,
#                                                                 len_after_states=num_states)  # domtools.
#     if rollout_cumu_dom_filter:
#         action_features = action_features[not_cumu_dominated]
#         map_back_vector = np.nonzero(not_cumu_dominated)[0]
#     else:
#         action_features = action_features[not_simply_dominated]
#         map_back_vector = np.nonzero(not_simply_dominated)[0]

# @njit(cache=False)
# def choose_action_in_rollout_with_filters(available_after_states, policy_weights,
#                                           num_features, reward_greedy,
#                                           rollout_dom_filter, rollout_cumu_dom_filter,
#                                           feature_directors):
#     num_states = len(available_after_states)
#     action_features = np.zeros((num_states, num_features))
#     for ix, after_state in enumerate(available_after_states):
#         action_features[ix] = after_state.get_features_pure(False)  # , order_by=self.feature_order
#     # if rollout_cumu_dom_filter or rollout_dom_filter:
#     not_simply_dominated, not_cumu_dominated = dominance_filter(action_features * feature_directors, len_after_states=num_states)  # domtools.
#     if rollout_cumu_dom_filter:
#         action_features = action_features[not_cumu_dominated]
#         map_back_vector = np.nonzero(not_cumu_dominated)[0]
#     else:
#         action_features = action_features[not_simply_dominated]
#         map_back_vector = np.nonzero(not_simply_dominated)[0]
#
#     utilities = action_features.dot(np.ascontiguousarray(policy_weights))
#     move_index = np.argmax(utilities)
#     move = available_after_states[map_back_vector[move_index]]
#     return move
#
#
#
# @njit(cache=False)
# def choose_action_in_rollout(available_after_states, policy_weights,
#                              num_features, reward_greedy):
#     num_states = len(available_after_states)
#     if reward_greedy:
#         rewards = np.zeros(num_states)
#         max_reward = 0
#         for ix, after_state in enumerate(available_after_states):
#             reward_of_after_state = after_state.n_cleared_lines
#             if reward_of_after_state > 0:
#                 rewards[ix] = after_state.n_cleared_lines
#                 if reward_of_after_state > max_reward:
#                     max_reward = reward_of_after_state
#         if max_reward > 0:
#             max_reward_indices = np.where(rewards == max_reward)[0]
#             available_after_states = [available_after_states[i] for i in max_reward_indices]
#             # action_features = action_features[max_reward_indeces]
#             num_states = len(available_after_states)
#     action_features = np.zeros((num_states, num_features))
#     for ix, after_state in enumerate(available_after_states):
#         action_features[ix] = after_state.get_features_pure(False)  # , order_by=self.feature_order
#     # if rollout_cumu_dom_filter:
#     #     not_simply_dominated, not_cumu_dominated = dom_filter(action_features, len_after_states=num_states)  # domtools.
#     #     action_features = action_features[not_cumu_dominated]
#     #     map_back_vector = np.nonzero(not_cumu_dominated)[0]
#     #     # if rollout_cumu_dom_filter:
#     #     #     available_after_states = available_after_states[not_simply_dominated]
#     #     #     action_features = action_features[not_simply_dominated]
#     #     # elif rollout_dom_filter:
#     #     #     available_after_states = available_after_states[not_cumu_dominated]
#     #     #     action_features = action_features[not_cumu_dominated]
#     # else:
#     #     raise ValueError("Currently only implemented with cumu_dom_filter")
#     utilities = action_features.dot(np.ascontiguousarray(policy_weights))
#     move_index = np.argmax(utilities)
#     move = available_after_states[move_index]
#     return move







