import numpy as np
from tetris.state import State, TerminalState
import numba
from numba import njit, float64, int64, bool_, int64
from numba.experimental import jitclass

spec_agent = [
    ('policy_weights', float64[:]),
    ('feature_directors', int64[:]),
    ('feature_type', numba.types.string),
    ('num_features', int64),
    ('feature_directors', float64[:]),
    ('uses_filters', bool_),
    ('direct_features', bool_)
]


@jitclass(spec_agent)
class ConstantAgent:
    #  feature_directors=np.array([-1, -1, -1, -1, -1, -1, 1, -1], dtype=np.float64)
    def __init__(self, policy_weights, feature_type="bcts",
                 feature_directors=np.array([-1, -1, -1, -1, -1, -1, 1, -1], dtype=np.float64),
                 use_dom_filter=False,
                 use_cumul_dom_filter=False):
        self.policy_weights = policy_weights
        self.feature_type = feature_type
        self.num_features = len(self.policy_weights)
        self.use_dom_filter = use_dom_filter
        self.use_cumul_dom_filter = use_cumul_dom_filter
        if self.use_dom_filter or self.use_cumul_dom_filter:  # or self.use_STEW
            self.uses_filters = True
            self.direct_features = True
        else:
            self.uses_filters = False
            self.direct_features = False

        assert self.feature_type == "bcts", "Features have to be 'bcts'."
        self.feature_directors = feature_directors

        # if np.all(feature_directors == np.ones(8)):
        #     print("Don't need directing")
        #     self.direct_features = False
        # else:
        #     print("Feature directors are being used.")
        #     self.direct_features = True

    # def choose_action(self, start_state, start_tetromino):
    #     """
    #     Chooses the utility-maximising action.
    #     """
    #     return move, move_index

    def choose_action(self, start_state, start_tetromino):
        if self.uses_filters:
            move = self.choose_action_test(start_state, start_tetromino)
        else:
            move = self.choose_action_test_with_filters(start_state, start_tetromino)
        return move

    def choose_action_test_with_filters(self, start_state, start_tetromino):
        """
        Chooses the utility-maximising action after dominance-filtering the action set.
        """
        children_states = start_tetromino.get_after_states(start_state)  # , current_state=
        num_children = len(children_states)
        if num_children == 0:
            # Terminal state!!
            return State(np.zeros((1, 1), dtype=np.bool_),
                         np.zeros(1, dtype=np.int64),
                         # changed_cols=np.array([0], dtype=np.int64),
                         np.array([0], dtype=np.int64),
                         np.array([0], dtype=np.int64),
                         0.0,
                         1,
                         "bcts",
                         True,
                         False)  #, move_index

        action_features = np.zeros((num_children, self.num_features))
        if self.direct_features:
            for ix, after_state in enumerate(children_states):
                action_features[ix] = after_state.get_features_and_direct(self.feature_directors, False)  # direct_by=self.feature_directors  , order_by=None  , addRBF=False
        else:
            for ix, after_state in enumerate(children_states):
                action_features[ix] = after_state.get_features_pure(False)  # direct_by=self.feature_directors  , order_by=None  , addRBF=False
        if self.direct_features:


        utilities = action_features.dot(np.ascontiguousarray(self.policy_weights))
        max_indices = np.where(utilities == np.max(utilities))[0]
        move_index = np.random.choice(max_indices)
        move = children_states[move_index]
        return move  #, move_index

    def choose_action_test(self, start_state, start_tetromino):
        # """
        # Chooses the utility-maximising action.
        # """
        children_states = start_tetromino.get_after_states(start_state)  # , current_state=
        num_children = len(children_states)
        if num_children == 0:
            # Terminal state!!
            return State(np.zeros((1, 1), dtype=np.bool_),
                         np.zeros(1, dtype=np.int64),
                         # changed_cols=np.array([0], dtype=np.int64),
                         np.array([0], dtype=np.int64),
                         np.array([0], dtype=np.int64),
                         0.0,
                         1,
                         "bcts",
                         True,
                         False)  #, move_index

        action_features = np.zeros((num_children, self.num_features))
        if self.direct_features:
            for ix, after_state in enumerate(children_states):
                action_features[ix] = after_state.get_features_and_direct(self.feature_directors, False)  # direct_by=self.feature_directors  , order_by=None  , addRBF=False
        else:
            for ix, after_state in enumerate(children_states):
                action_features[ix] = after_state.get_features_pure(False)  # direct_by=self.feature_directors  , order_by=None  , addRBF=False
        utilities = action_features.dot(np.ascontiguousarray(self.policy_weights))
        max_indices = np.where(utilities == np.max(utilities))[0]
        move_index = np.random.choice(max_indices)
        move = children_states[move_index]
        return move  #, move_index
