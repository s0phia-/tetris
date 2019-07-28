import numpy as np
from tetris.state import State, TerminalState
from numba import njit


class ConstantAgent:
    def __init__(self, policy_weights, feature_type="bcts", feature_directors=None):
        self.policy_weights = policy_weights
        self.feature_type = feature_type
        self.num_features = len(self.policy_weights)
        if feature_directors is None:
            if self.feature_type == "bcts":
                print("Features are directed automatically to be BCTS features.")
                self.feature_directors = np.array([-1, -1, -1, -1, -1, -1, 1, -1])
        else:
            self.feature_directors = feature_directors

    # def choose_action(self, start_state, start_tetromino):
    #     """
    #     Chooses the utility-maximising action.
    #     """
    #     children_states = start_tetromino.get_after_states(start_state) #, current_state=
    #     # available_after_states = np.array([child for child in all_available_after_states if not child.terminal_state])
    #     num_children = len(children_states)
    #     if num_children == 0:
    #         # Game over!
    #         return TerminalState(), 0
    #     action_features = np.zeros((num_children, self.num_features))
    #     for ix, after_state in enumerate(children_states):
    #         action_features[ix] = after_state.get_features(direct_by=self.feature_directors)  # , order_by=None , addRBF=False
    #     utilities = action_features.dot(self.policy_weights)
    #     max_indices = np.where(utilities == np.max(utilities))[0]
    #     move_index = np.random.choice(max_indices)
    #     move = children_states[move_index]
    #     return move, move_index

    def choose_action_test(self, start_state, start_tetromino):
        # """
        # Chooses the utility-maximising action.
        # """
        # children_states = start_tetromino.get_after_states(start_state)  # , current_state=
        # # available_after_states = np.array([child for child in all_available_after_states if not child.terminal_state])
        # num_children = len(children_states)
        # if num_children == 0:
        #     # print("Game over!")
        #     return TerminalState(), 0
        # action_features = np.zeros((num_children, self.num_features))
        # for ix, after_state in enumerate(children_states):
        #     action_features[ix] = after_state.get_features(self.feature_directors, False)  # direct_by=self.feature_directors  , order_by=None  , addRBF=False
        # utilities = action_features.dot(self.policy_weights)
        # max_indices = np.where(utilities == np.max(utilities))[0]
        # move_index = np.random.choice(max_indices)
        # move = children_states[move_index]
        # return move, move_index
        return chosen_action_jitted(start_state, start_tetromino,
                                    self.num_features, self.feature_directors, self.policy_weights)


@njit
def chosen_action_jitted(start_state, start_tetromino, num_features, feature_directors, policy_weights):
    # TODO: remove after testing
    # np.random.seed(1)
    children_states = start_tetromino.get_after_states(start_state)  # , current_state=
    # available_after_states = np.array([child for child in all_available_after_states if not child.terminal_state])
    num_children = len(children_states)
    if num_children == 0:
        # print("Game over!")
        # return "terminal state"
        return State(np.zeros((1, 1), dtype=np.bool_),
                     np.zeros(1, dtype=np.int8),
                     # changed_cols=np.array([0], dtype=np.int8),
                     np.array([0], dtype=np.int8),
                     np.array([0], dtype=np.int8),
                     0.0,
                     1,
                     "bcts",
                     True)

    action_features = np.zeros((num_children, num_features))
    for ix, after_state in enumerate(children_states):
        action_features[ix] = after_state.get_features(feature_directors, False)  # direct_by=self.feature_directors  , order_by=None  , addRBF=False
    utilities = action_features.dot(policy_weights)
    max_indices = np.where(utilities == np.max(utilities))[0]
    move_index = np.random.choice(max_indices)
    move = children_states[move_index]
    return move
