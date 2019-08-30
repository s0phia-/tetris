import numpy as np

from tetris import state, tetromino
from tetris.utils import print_board_to_string, print_tetromino

np.random.seed(3)

@njit(fastmath=True, cache=False)
def calc_lowest_free_rows(rep):
    num_rows, n_cols = rep.shape
    lowest_free_rows = np.zeros(n_cols, dtype=np.int64)
    for col_ix in range(n_cols):
        lowest = 0
        for row_ix in range(num_rows - 1, -1, -1):
            if rep[row_ix, col_ix] == 1:
                lowest = row_ix + 1
                break
        lowest_free_rows[col_ix] = lowest
    return lowest_free_rows


representation = np.array([[1, 0, 1, 0, 0, 1, 0, 1, 1, 1],
                           [0, 0, 0, 1, 1, 1, 0, 1, 1, 1],
                           [1, 1, 1, 1, 1, 0, 0, 1, 1, 1],
                           [0, 1, 1, 1, 1, 1, 0, 1, 1, 1],
                           [0, 1, 1, 1, 1, 1, 0, 1, 1, 1],
                           [0, 1, 0, 1, 1, 1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1, 0, 1, 1, 1, 1],
                           [1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                           [1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
                           [0, 0, 0, 0, 0, 0, 1, 0, 1, 1]])


lowest_free_rows = calc_lowest_free_rows(representation)

st = state.State(representation=representation, lowest_free_rows=lowest_free_rows,
                 changed_lines=np.array([0], dtype=np.int64),
                 pieces_per_changed_row=np.array([0], dtype=np.int64),
                 landing_height_bonus=0.0,
                 num_features=8,
                 feature_type="bcts",
                 terminal_state=False,  # this is useful to generate a "terminal state"
                 has_overlapping_fields=False)

print(print_board_to_string(st))

tet = tetromino.Tetromino(feature_type="bcts", num_features=8, num_columns=10)
print(print_tetromino(tet.current_tetromino))

af_st = tet.get_after_states(st)
print(print_board_to_string(af_st[0]))

