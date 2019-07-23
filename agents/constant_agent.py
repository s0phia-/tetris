import numpy as np
from tetris.state import TerminalState


class ConstantAgent:
    def __init__(self, policy_weights, feature_type="bcts", feature_directors=None):
        self.policy_weights = policy_weights
        self.feature_type = feature_type
        self.num_features = len(self.policy_weights)
        if feature_directors is None:
            if self.feature_type == 'bcts':
                print("Features are directed automatically to be BCTS features.")
                self.feature_directors = np.array([-1, -1, -1, -1, -1, -1, 1, -1])
        else:
            self.feature_directors = feature_directors

    def choose_action(self, start_state, start_tetromino):
        """
        Chooses the utility-maximising action.
        """
        children_states = start_tetromino.get_after_states(current_state=start_state)
        # available_after_states = np.array([child for child in all_available_after_states if not child.terminal_state])
        num_children = len(children_states)
        if num_children == 0:
            # Game over!
            return TerminalState(), 0
        action_features = np.zeros((num_children, self.num_features))
        for ix, after_state in enumerate(children_states):
            action_features[ix] = after_state.get_features(direct_by=self.feature_directors, order_by=None)
        utilities = action_features.dot(self.policy_weights)
        max_indices = np.where(utilities == np.max(utilities))[0]
        move_index = np.random.choice(max_indices)
        move = children_states[move_index]
        return move, move_index

    def choose_action_test(self, start_state, start_tetromino):
        return self.choose_action(start_state, start_tetromino)
