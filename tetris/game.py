import numpy as np
from tetris import state, tetromino
import collections
import time
from tetris.utils import print_board_to_string



class Tetris:
    def __init__(self, num_columns, num_rows, verbose=False,
                 plot_intermediate_results=False, feature_type='bcts', num_features=8,
                 tetromino_size=4, target_update=1, max_cleared_test_lines=np.inf):
        self.afterstates = None
        self.num_columns = num_columns
        self.num_rows = num_rows
        self.tetromino_size = tetromino_size
        # self.player = player
        self.verbose = verbose
        self.target_update = target_update
        self.num_features = num_features
        self.feature_type = feature_type
        self.n_fields = self.num_columns * self.num_rows
        self.game_over = False
        self.current_state = state.State(
            representation=np.zeros((self.num_rows + self.tetromino_size, self.num_columns), dtype=np.int_),
            lowest_free_rows=np.zeros(self.num_columns, dtype=np.int_), num_features=num_features,
            feature_type=feature_type)
        self.tetrominos = [tetromino.Straight(feature_type, num_features, self.num_columns),
                           tetromino.RCorner(feature_type, num_features, self.num_columns),
                           tetromino.LCorner(feature_type, num_features, self.num_columns),
                           tetromino.Square(feature_type, num_features, self.num_columns),
                           tetromino.SnakeR(feature_type, num_features, self.num_columns),
                           tetromino.SnakeL(feature_type, num_features, self.num_columns),
                           tetromino.T(feature_type, num_features, self.num_columns)]
        self.tetromino_sampler = tetromino.TetrominoSamplerRandom(self.tetrominos)
        self.cleared_lines = 0
        self.state_samples = []
        self.cumulative_steps = 0
        self.max_cleared_test_lines = max_cleared_test_lines
        self.plot_intermediate_results = plot_intermediate_results
        self.current_tetromino = None

    def reset(self):
        self.game_over = False
        self.current_state = state.State(
            representation=np.zeros((self.num_rows + self.tetromino_size, self.num_columns), dtype=np.int_),
            lowest_free_rows=np.zeros(self.num_columns, dtype=np.int_), num_features=self.num_features,
            feature_type=self.feature_type)
        self.tetromino_sampler = tetromino.TetrominoSampler(self.tetrominos)
        self.cleared_lines = 0
        self.state_samples = []
        self.current_tetromino = self.tetromino_sampler.next_tetromino()
        self.afterstates = None

    def get_after_states(self):
        afterstates = self.current_tetromino.get_after_states(self.current_state)  # the actions are afterstates
        available_after_states = np.array([child for child in afterstates if not child.terminal_state])
        if len(available_after_states) == 0:
            self.game_over = True
        num_states = len(available_after_states)
        action_features = np.zeros((num_states, self.num_features))
        for ix, after_state in enumerate(available_after_states):
            action_features[ix] = after_state.get_features()  # can use directions here
        self.afterstates = available_after_states
        return action_features

    def step(self, action_ix):
        self.get_after_states()
        observation = self.afterstates[action_ix]
        self.cleared_lines += observation.n_cleared_lines
        reward = observation.n_cleared_lines  # Malte used self.cleared_lines for this
        self.current_state = observation
        self.is_game_over()
        done = self.game_over
        info = None
        self.current_tetromino = self.tetromino_sampler.next_tetromino()
        return observation, reward, done, info

    def is_game_over(self):
        afterstates = self.current_tetromino.get_after_states(self.current_state)  # the actions are afterstates
        available_after_states = np.array([child for child in afterstates if not child.terminal_state])
        if len(available_after_states) == 0:
            self.game_over = True
            return True

    def print_current_board(self):
        print(print_board_to_string(self.current_state))

    def print_current_tetromino(self):
        print(self.current_tetromino.__repr__())
