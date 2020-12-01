import numpy as np
from tetris.state import State, TerminalState
import numba
from numba import njit, float64, int64, bool_, int64
from numba.experimental import jitclass
from domtools import dom_filter as dominance_filter

spec_agent = [
    ('policy_weights', float64[:]),
    ('feature_type', numba.types.string),
    ('num_features', int64),
    ('feature_directors', float64[:]),
    ('use_filter_in_eval', bool_),
    ('use_dom_filter', bool_),
    ('use_cumul_dom_filter', bool_)
]


@jitclass(spec_agent)
class ConstantAgent:
    def __init__(self, policy_weights, feature_type="bcts",
                 feature_directors=np.array([-1, -1, -1, -1, -1, -1, 1, -1], dtype=np.float64),
                 use_filter_in_eval=False,
                 use_dom_filter=False,
                 use_cumul_dom_filter=False):
        self.policy_weights = policy_weights
        self.feature_type = feature_type
        self.num_features = len(self.policy_weights)
        self.use_filter_in_eval = use_filter_in_eval
        if self.use_filter_in_eval:
            # Only set use_dom_filter and use_cumul_dom_filter if use_filter_in_eval == True
            # (use_dom_filter and use_cumul_dom_filter) decide globally "simple vs. cumul" dominance.
            # _WHERE_ to use filters is decided by `use_filter_in_eval` and `use_filters_during_rollout`
            assert use_dom_filter or use_cumul_dom_filter and not (use_dom_filter and use_cumul_dom_filter)
            self.use_dom_filter = use_dom_filter
            self.use_cumul_dom_filter = use_cumul_dom_filter
            # self.choose_action_test = self.choose_action_test_with_filters
        else:
            self.use_dom_filter = False
            self.use_cumul_dom_filter = False
            # self.choose_action_test = self.choose_action_test_without_filters

        assert self.feature_type == "bcts", "Features have to be 'bcts'."
        self.feature_directors = feature_directors

    def choose_action(self, start_state, start_tetromino):
        children_states = start_tetromino.get_after_states(start_state)  # , current_state=
        num_children = len(children_states)
        if num_children == 0:
            # Terminal state!!
            return State(np.zeros((1, 1), dtype=np.bool_),
                         np.zeros(1, dtype=np.int64),
                         np.array([0], dtype=np.int64),
                         np.array([0], dtype=np.int64),
                         0.0,
                         1,
                         "bcts",
                         True,
                         False)

        action_features = np.zeros((num_children, self.num_features))
        for ix, after_state in enumerate(children_states):
            action_features[ix] = after_state.get_features_pure(False)

        if self.use_filter_in_eval:
            not_simply_dominated, not_cumu_dominated = dominance_filter(action_features * self.feature_directors,
                                                                        len_after_states=num_children)  # domtools.
            if self.use_cumul_dom_filter:
                action_features = action_features[not_cumu_dominated]
                map_back_vector = np.nonzero(not_cumu_dominated)[0]
            else:
                action_features = action_features[not_simply_dominated]
                map_back_vector = np.nonzero(not_simply_dominated)[0]

        utilities = action_features.dot(np.ascontiguousarray(self.policy_weights))
        max_indices = np.where(utilities == np.max(utilities))[0]
        move_index = np.random.choice(max_indices)
        if self.use_filter_in_eval:
            move = children_states[map_back_vector[move_index]]
        else:
            move = children_states[move_index]
        return move

    # def choose_action_test_with_filters(self, start_state, start_tetromino):
    #     """
    #     Chooses the utility-maximising action after dominance-filtering the action set.
    #     """
    #     children_states = start_tetromino.get_after_states(start_state)  # , current_state=
    #     num_children = len(children_states)
    #     if num_children == 0:
    #         # Terminal state!!
    #         return State(np.zeros((1, 1), dtype=np.bool_),
    #                      np.zeros(1, dtype=np.int64),
    #                      np.array([0], dtype=np.int64),
    #                      np.array([0], dtype=np.int64),
    #                      0.0,
    #                      1,
    #                      "bcts",
    #                      True,
    #                      False)
    #
    #     action_features = np.zeros((num_children, self.num_features))
    #     for ix, after_state in enumerate(children_states):
    #         action_features[ix] = after_state.get_features_pure(False)
    #
    #     not_simply_dominated, not_cumu_dominated = dominance_filter(action_features * self.feature_directors,
    #                                                                 len_after_states=num_children)  # domtools.
    #     if self.use_cumul_dom_filter:
    #         action_features = action_features[not_cumu_dominated]
    #         map_back_vector = np.nonzero(not_cumu_dominated)[0]
    #     else:
    #         action_features = action_features[not_simply_dominated]
    #         map_back_vector = np.nonzero(not_simply_dominated)[0]
    #
    #     utilities = action_features.dot(np.ascontiguousarray(self.policy_weights))
    #     max_indices = np.where(utilities == np.max(utilities))[0]
    #     move_index = np.random.choice(max_indices)
    #     move = children_states[map_back_vector[move_index]]
    #     return move
    #
    # def choose_action_test_without_filters(self, start_state, start_tetromino):
    #     # """
    #     # Chooses the utility-maximising action.
    #     # """
    #     children_states = start_tetromino.get_after_states(start_state)  # , current_state=
    #     num_children = len(children_states)
    #     if num_children == 0:
    #         # Terminal state!!
    #         return State(np.zeros((1, 1), dtype=np.bool_),
    #                      np.zeros(1, dtype=np.int64),
    #                      np.array([0], dtype=np.int64),
    #                      np.array([0], dtype=np.int64),
    #                      0.0,
    #                      1,
    #                      "bcts",
    #                      True,
    #                      False)
    #
    #     action_features = np.zeros((num_children, self.num_features))
    #
    #     for ix, after_state in enumerate(children_states):
    #         action_features[ix] = after_state.get_features_pure(False)
    #     utilities = action_features.dot(np.ascontiguousarray(self.policy_weights))
    #     max_indices = np.where(utilities == np.max(utilities))[0]
    #     move_index = np.random.choice(max_indices)
    #     move = children_states[move_index]
    #     return move
