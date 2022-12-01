import numpy as np
from tetris import state
from tetris import tetromino
from tetris.utils import print_board_to_string


class Tetris:
    def __init__(self, num_columns, num_rows, feature_directions=None,
                 feature_type='bcts', num_features=8,
                 tetromino_size=4):

        # Saving parameters
        self.feature_directions = feature_directions
        self.num_columns = num_columns
        self.num_rows = num_rows
        self.feature_type = feature_type
        self.num_features = num_features
        self.tetromino_size = tetromino_size

        # The tetronimoes
        self.tetrominos = [tetromino.Straight(feature_type, num_features, self.num_columns),
                           tetromino.RCorner(feature_type, num_features, self.num_columns),
                           tetromino.LCorner(feature_type, num_features, self.num_columns),
                           tetromino.Square(feature_type, num_features, self.num_columns),
                           tetromino.SnakeR(feature_type, num_features, self.num_columns),
                           tetromino.SnakeL(feature_type, num_features, self.num_columns),
                           tetromino.T(feature_type, num_features, self.num_columns)]

        # game setup
        self.tetromino_sampler = tetromino.TetrominoSampler(self.tetrominos)
        self.reset()

    def reset(self):
        # Reset state and choose first tetronimno
        self.current_state = state.State(
            representation=np.zeros((self.num_rows + self.tetromino_size, self.num_columns), dtype=np.int_),
            lowest_free_rows=np.zeros(self.num_columns, dtype=np.int_), num_features=self.num_features,
            feature_type=self.feature_type)

        self.current_tetromino = self.tetromino_sampler.next_tetromino()

        # return np.append(self.current_state.representation.flatten(), self.current_tetromino.tet_ind)

    def get_after_states(self, include_terminal=False):
        afterstates = self.current_tetromino.get_after_states(self.current_state)  # the actions are afterstates
        self.afterstates = np.array([child for child in afterstates if not child.terminal_state])
        action_features = np.zeros((len(self.afterstates), self.num_features))
        for ix, after_state in enumerate(self.afterstates):
            action_features[ix] = after_state.get_features(direct_by=self.feature_directions)  # can use directions here
        # todo
        if include_terminal:
            all_afterstates = np.zeros((len(afterstates), self.num_features))
            for ix, after_state in enumerate(afterstates):
                all_afterstates[ix] = after_state.get_features(direct_by=self.feature_directions)  # can use directions here
            return action_features, all_afterstates
        else:
            return action_features

    def step(self, action):
        observation_features = self.get_after_states()[action]
        self.current_state = self.afterstates[action]
        reward = self.current_state.n_cleared_lines  # Malte used self.cleared_lines for this
        self.current_tetromino = self.tetromino_sampler.next_tetromino()
        done = self.is_game_over(self.current_state)
        return observation_features, reward, done, None

    def is_game_over(self, state):
        afterstates = self.current_tetromino.get_after_states(state)
        available_after_states = np.array([child for child in afterstates if not child.terminal_state])
        if len(available_after_states) == 0:
            return True
        else:
            return False

    def get_best_policy(self):

        after_states = self.current_tetromino.get_after_states(self.current_state)

        after_state_fitness = np.array(list(map(self.fitness, after_states)))

        best_policy = (after_state_fitness == after_state_fitness.max()).astype(float)
        best_policy /= best_policy.sum()

        return best_policy

    def fitness(self, state):

        state_features = state.get_features()

        fitness_value = state_features[0] * -24.04 + \
                        state_features[1] * -19.77 + \
                        state_features[2] * -13.08 + \
                        state_features[3] * -12.63 + \
                        state_features[4] * -10.49 + \
                        state_features[5] * -9.22 + \
                        state_features[6] * 6.6 + \
                        state_features[7] * -1.61

        return fitness_value

    def render(self):

        print(print_board_to_string(self.current_state))
        print(self.current_tetromino)

    def get_current_state_features(self):
        return self.current_state.get_features(direct_by=self.feature_directions)
