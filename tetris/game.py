import numpy as np
from tetris import state, tetromino  # tetromino_old,
import numba
from numba import jitclass, bool_, int64


specTetris = [
    ('num_columns', int64),
    ('num_rows', int64),
    ('vebose', bool_),
    ('tetromino_size', int64),
    ('feature_type', numba.types.string),
    ('num_features', int64),
    ('max_cleared_test_lines', int64),
    ('game_over', bool_),
    ('current_state', state.State.class_type.instance_type),
    ('generative_model', tetromino.Tetromino.class_type.instance_type),
    ('cleared_lines', int64)
]


@jitclass(specTetris)
class Tetris:
    """
    Tetris for reinforcement learning applications.

    Tailored to use with a set of hand-crafted features such as "BCTS" (Thiery & Scherrer 2009)

    The BCTS feature names (and order) are
    ['rows_with_holes', 'column_transitions', 'holes', 'landing_height',
    'cumulative_wells', 'row_transitions', 'eroded', 'hole_depth']

    """
    def __init__(self,
                 num_columns,
                 num_rows,
                 max_cleared_test_lines=10e9,
                 tetromino_size=4,
                 feature_type="bcts",
                 num_features=8
                 ):
        """
        
        :param num_columns: 
        :param num_rows:
        :param tetromino_size:
        :param max_cleared_test_lines:
        """
        self.num_columns = num_columns
        self.num_rows = num_rows
        self.tetromino_size = tetromino_size
        self.num_features = num_features
        self.feature_type = feature_type
        self.max_cleared_test_lines = max_cleared_test_lines
        self.game_over = False
        self.current_state = state.State(np.zeros((self.num_rows, self.num_columns), dtype=np.bool_),  # representation=
                                         np.zeros(self.num_columns, dtype=np.int64),  # lowest_free_rows=
                                         np.array([0], dtype=np.int64),  # changed_lines=
                                         np.array([0], dtype=np.int64),  # pieces_per_changed_row=
                                         0.0,  # landing_height_bonus=
                                         self.num_features,  # num_features=
                                         "bcts",  # feature_type=
                                         False,  # terminal_state=
                                         False  # has_overlapping_fields=
                                         )
        self.generative_model = tetromino.Tetromino(self.feature_type, self.num_features, self.num_columns)
        self.cleared_lines = 0

    def reset(self):
        self.game_over = False
        self.current_state = state.State(np.zeros((self.num_rows, self.num_columns), dtype=np.bool_),  # representation=
                                         np.zeros(self.num_columns, dtype=np.int64),  # lowest_free_rows=
                                         np.array([0], dtype=np.int64),  # changed_lines=
                                         np.array([0], dtype=np.int64),  # pieces_per_changed_row=
                                         0.0,  # landing_height_bonus=
                                         self.num_features,  # num_features=
                                         "bcts",  # feature_type=
                                         False,  # terminal_state=
                                         False  # has_overlapping_fields=
                                         )
        self.current_state.calc_bcts_features()
        self.cleared_lines = 0
        self.generative_model.next_tetromino()

    def make_step(self, after_state):
        self.game_over = after_state.terminal_state
        if not self.game_over:
            self.cleared_lines += after_state.n_cleared_lines
            self.current_state = after_state
            self.generative_model.next_tetromino()

