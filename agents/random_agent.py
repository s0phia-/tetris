import numpy as np


class RandomAgent:
    def __init__(self):
        pass

    def choose_action(self, start_state, start_tetromino):
        all_children_states = start_tetromino.get_after_states(current_state=start_state)
        # TODO: can speed this up maybe? Only put in non-terminal-states?
        children_states = np.array([child for child in all_children_states if not child.terminal_state])
        if len(children_states) == 0:
            # Game over!
            return np.random.choice(all_children_states)
        else:
            return np.random.choice(children_states)

    def choose_action_test(self, start_state, start_tetromino):
        return self.choose_action(start_state, start_tetromino)
