# import numpy as np
# import numba
# from numba import njit, jitclass, float64, int64, bool_
# from tetris import state
# from numba.typed import List
#
#
# # TODO: introduce typed list
#
#
# specT = [
#     ('feature_type', numba.types.string),
#     ('num_features', int64),
#     ('num_columns', int64)
# ]
#
#
# class TetrominoSamplerRandom:
#     def __init__(self, tetrominos):
#         self.tetrominos = tetrominos
#
#     def next_tetromino(self):
#         return np.random.choice(a=self.tetrominos, size=1)[0]
#
#
#
# @jitclass(specT)
# class Straight:
#     def __init__(self, feature_type, num_features, num_columns):
#         # Tetromino.__init__(self, feature_type, num_features, num_columns)
#         self.feature_type = feature_type
#         self.num_features = num_features
#         self.num_columns = num_columns
#
# #     def __repr__(self):
# #         return '''
# # ██ ██ ██ ██'''
#
#     def get_after_states(self, current_state):
#         # after_states = List()
#         after_states = []
#         # Vertical placements
#         for col_ix, free_pos in enumerate(current_state.lowest_free_rows):
#             anchor_row = free_pos
#             if not anchor_row + 4 > current_state.num_rows:
#                 new_lowest_free_rows = current_state.lowest_free_rows.copy()
#                 new_lowest_free_rows[col_ix] += 4
#                 new_representation = current_state.representation.copy()
#                 new_representation[anchor_row:(anchor_row + 4), col_ix] = 1
#                 new_state = state.State(new_representation,
#                                         new_lowest_free_rows,
#                                         # np.array([col_ix], dtype=np.int64),
#                                         np.arange(anchor_row, anchor_row + 4, 1, np.int64),
#                                         np.array([1, 1, 1, 1], dtype=np.int64),
#                                         1.5,
#                                         self.num_features,
#                                         self.feature_type,
#                                         False)
#
#                                         # current_state.col_transitions_per_col,
#                                         # current_state.row_transitions_per_col,
#                                         # current_state.array_of_rows_with_holes,
#                                         # current_state.holes_per_col,
#                                         # current_state.hole_depths_per_col,
#                                         # current_state.cumulative_wells_per_col)
#                 after_states.append(new_state)
#
#             # Horizontal placements
#             if col_ix < self.num_columns - 3:
#             # max_col_index = self.num_columns - 3
#             # for col_ix, free_pos in enumerate(current_state.lowest_free_rows[:max_col_index]):
#                 anchor_row = np.max(current_state.lowest_free_rows[col_ix:(col_ix + 4)])
#                 if not anchor_row + 1 > current_state.num_rows:
#                     new_lowest_free_rows = current_state.lowest_free_rows.copy()
#                     new_lowest_free_rows[col_ix:(col_ix + 4)] = anchor_row + 1
#                     new_representation = current_state.representation.copy()
#                     new_representation[anchor_row, col_ix:(col_ix + 4)] = 1
#                     new_state = state.State(new_representation,
#                                             new_lowest_free_rows,
#                                             # np.arange(col_ix, col_ix + 4, 1, np.int64),
#                                             np.arange(anchor_row, anchor_row + 1, 1, np.int64),
#                                             np.array([4], dtype=np.int64),
#                                             0,
#                                             self.num_features,
#                                             self.feature_type,
#                                             False)
#                                             # current_state.col_transitions_per_col,
#                                             # current_state.row_transitions_per_col,
#                                             # current_state.array_of_rows_with_holes,
#                                             # current_state.holes_per_col,
#                                             # current_state.hole_depths_per_col,
#                                             # current_state.cumulative_wells_per_col)
#                     after_states.append(new_state)
#         return after_states
#
#
# @jitclass(specT)
# class Square:
#     def __init__(self, feature_type, num_features, num_columns):
#         self.feature_type = feature_type
#         self.num_features = num_features
#         self.num_columns = num_columns
#
# #     def __repr__(self):
# #         return '''
# # ██ ██
# # ██ ██'''
#
#     def get_after_states(self, current_state):
#         # after_states = List()
#         after_states = []
#         # Horizontal placements
#         max_col_index = self.num_columns - 1
#         for col_ix, free_pos in enumerate(current_state.lowest_free_rows[:max_col_index]):
#             anchor_row = np.max(current_state.lowest_free_rows[col_ix:(col_ix + 2)])
#             if not anchor_row + 2 > current_state.num_rows:
#                 new_lowest_free_rows = current_state.lowest_free_rows.copy()
#                 new_lowest_free_rows[col_ix:(col_ix + 2)] = anchor_row + 2
#                 new_representation = current_state.representation.copy()
#                 new_representation[anchor_row:(anchor_row + 2), col_ix:(col_ix + 2)] = 1
#                 new_state = state.State(new_representation,
#                                         new_lowest_free_rows,
#                                         # np.arange(col_ix, col_ix + 2, 1, np.int64),
#                                         np.arange(anchor_row, anchor_row + 2, 1, np.int64),
#                                         np.array([2, 2], dtype=np.int64),
#                                         0.5,
#                                         self.num_features,
#                                         self.feature_type,
#                                         False)
#                                         # current_state.col_transitions_per_col,
#                                         # current_state.row_transitions_per_col,
#                                         # current_state.array_of_rows_with_holes,
#                                         # current_state.holes_per_col,
#                                         # current_state.hole_depths_per_col,
#                                         # current_state.cumulative_wells_per_col
#                                         # )
#                 after_states.append(new_state)
#         return after_states
#
#
# @jitclass(specT)
# class SnakeR:
#     def __init__(self, feature_type, num_features, num_columns):
#         self.feature_type = feature_type
#         self.num_features = num_features
#         self.num_columns = num_columns
#
#
# #     def __repr__(self):
# #         return '''
# #    ██ ██
# # ██ ██'''
#
#     def get_after_states(self, current_state):
#         # after_states = List()
#         after_states = []
#         # Vertical placements
#         max_col_index = self.num_columns - 1
#         for col_ix, free_pos in enumerate(current_state.lowest_free_rows[:max_col_index]):
#             anchor_row = np.maximum(current_state.lowest_free_rows[col_ix] - 1, current_state.lowest_free_rows[col_ix + 1])
#             if not anchor_row + 3 > current_state.num_rows:
#                 new_lowest_free_rows = current_state.lowest_free_rows.copy()
#                 new_lowest_free_rows[col_ix] = anchor_row + 3
#                 new_lowest_free_rows[col_ix + 1] = anchor_row + 2
#                 new_representation = current_state.representation.copy()
#                 new_representation[(anchor_row + 1):(anchor_row + 3), col_ix] = 1
#                 new_representation[anchor_row:(anchor_row + 2), col_ix + 1] = 1
#                 new_state = state.State(new_representation,
#                                         new_lowest_free_rows,
#                                         # np.arange(col_ix, col_ix + 2, 1, np.int64),
#                                         np.arange(anchor_row, anchor_row + 2, 1, np.int64),
#                                         np.array([1, 2], dtype=np.int64),
#                                         1,
#                                         self.num_features,
#                                         self.feature_type,
#                                         False)
#
#                                                                                     # current_state.col_transitions_per_col,
#                                             # current_state.row_transitions_per_col,
#                                             # current_state.array_of_rows_with_holes,
#                                             # current_state.holes_per_col,
#                                             # current_state.hole_depths_per_col,
#                                             # current_state.cumulative_wells_per_col)
#                 after_states.append(new_state)
#
#             # Horizontal placements
#             # max_col_index = self.num_columns - 2
#             # for col_ix, free_pos in enumerate(current_state.lowest_free_rows[:max_col_index]):
#             if col_ix < self.num_columns - 2:
#                 anchor_row = np.maximum(np.max(current_state.lowest_free_rows[col_ix:(col_ix + 2)]), current_state.lowest_free_rows[col_ix + 2] - 1)
#                 if not anchor_row + 2 > current_state.num_rows:
#                     new_lowest_free_rows = current_state.lowest_free_rows.copy()
#                     new_lowest_free_rows[col_ix] = anchor_row + 1
#                     new_lowest_free_rows[(col_ix + 1):(col_ix + 3)] = anchor_row + 2
#                     new_representation = current_state.representation.copy()
#                     new_representation[anchor_row, col_ix:(col_ix + 2)] = 1
#                     new_representation[anchor_row + 1, (col_ix + 1):(col_ix + 3)] = 1
#                     new_state = state.State(new_representation,
#                                             new_lowest_free_rows,
#                                             # np.arange(col_ix, col_ix + 3, 1, np.int64),
#                                             np.arange(anchor_row, anchor_row + 1, 1, np.int64),
#                                             np.array([2], dtype=np.int64),
#                                             0.5,
#                                             self.num_features,
#                                             self.feature_type,
#                                             False)
#
#                                                                                         # current_state.col_transitions_per_col,
#                                             # current_state.row_transitions_per_col,
#                                             # current_state.array_of_rows_with_holes,
#                                             # current_state.holes_per_col,
#                                             # current_state.hole_depths_per_col,
#                                             # current_state.cumulative_wells_per_col)
#                     after_states.append(new_state)
#         return after_states
#
#
# @jitclass(specT)
# class SnakeL:
#     def __init__(self, feature_type, num_features, num_columns):
#         self.feature_type = feature_type
#         self.num_features = num_features
#         self.num_columns = num_columns
#
# #     def __repr__(self):
# #         return '''
# # ██ ██
# #    ██ ██'''
#
#     def get_after_states(self, current_state):
#         # after_states = List()
#         after_states = []
#         # Vertical placements
#         max_col_index = self.num_columns - 1
#         for col_ix, free_pos in enumerate(current_state.lowest_free_rows[:max_col_index]):
#             anchor_row = np.maximum(current_state.lowest_free_rows[col_ix], current_state.lowest_free_rows[col_ix + 1] - 1)
#             if not anchor_row + 3 > current_state.num_rows:
#                 new_lowest_free_rows = current_state.lowest_free_rows.copy()
#                 new_lowest_free_rows[col_ix] = anchor_row + 2
#                 new_lowest_free_rows[col_ix + 1] = anchor_row + 3
#                 new_representation = current_state.representation.copy()
#                 new_representation[anchor_row:(anchor_row + 2), col_ix] = 1
#                 new_representation[(anchor_row + 1):(anchor_row + 3), col_ix + 1] = 1
#                 new_state = state.State(new_representation,
#                                         new_lowest_free_rows,
#                                         # np.arange(col_ix, col_ix + 2, 1, np.int64),
#                                         np.arange(anchor_row, anchor_row + 2, 1, np.int64),
#                                         np.array([1, 2], dtype=np.int64),
#                                         1,
#                                         self.num_features,
#                                         self.feature_type,
#                                         False)
#
#                                                                                     # current_state.col_transitions_per_col,
#                                             # current_state.row_transitions_per_col,
#                                             # current_state.array_of_rows_with_holes,
#                                             # current_state.holes_per_col,
#                                             # current_state.hole_depths_per_col,
#                                             # current_state.cumulative_wells_per_col)
#                 after_states.append(new_state)
#
#             ## Horizontal placements
#             # max_col_index = self.num_columns - 2
#             # for col_ix, free_pos in enumerate(current_state.lowest_free_rows[:max_col_index]):
#             if col_ix < self.num_columns - 2:
#                 anchor_row = np.maximum(current_state.lowest_free_rows[col_ix] - 1, np.max(current_state.lowest_free_rows[(col_ix + 1):(col_ix + 3)]))
#                 if not anchor_row + 2 > current_state.num_rows:
#                     new_lowest_free_rows = current_state.lowest_free_rows.copy()
#                     new_lowest_free_rows[col_ix:(col_ix + 2)] = anchor_row + 2
#                     new_lowest_free_rows[col_ix + 2] = anchor_row + 1
#                     new_representation = current_state.representation.copy()
#                     new_representation[anchor_row, (col_ix + 1):(col_ix + 3)] = 1
#                     new_representation[anchor_row + 1, col_ix:(col_ix + 2)] = 1
#                     new_state = state.State(new_representation,
#                                             new_lowest_free_rows,
#                                             # np.arange(col_ix, col_ix + 3, 1, np.int64),
#                                             np.arange(anchor_row, anchor_row + 1, 1, np.int64),
#                                             np.array([2], dtype=np.int64),
#                                             0.5,
#                                             self.num_features,
#                                             self.feature_type,
#                                             False)
#
#                                                                                         # current_state.col_transitions_per_col,
#                                             # current_state.row_transitions_per_col,
#                                             # current_state.array_of_rows_with_holes,
#                                             # current_state.holes_per_col,
#                                             # current_state.hole_depths_per_col,
#                                             # current_state.cumulative_wells_per_col)
#                     after_states.append(new_state)
#         return after_states
#
#
# @jitclass(specT)
# class T:
#     def __init__(self, feature_type, num_features, num_columns):
#         self.feature_type = feature_type
#         self.num_features = num_features
#         self.num_columns = num_columns
#
# #     def __repr__(self):
# #         return """
# #    ██
# # ██ ██ ██"""
#
#     def get_after_states(self, current_state):
#         # after_states = List()
#         after_states = []
#         # Vertical placements.
#         max_col_index = self.num_columns - 1
#         for col_ix, free_pos in enumerate(current_state.lowest_free_rows[:max_col_index]):
#             # Single cell on left
#             anchor_row = np.maximum(current_state.lowest_free_rows[col_ix] - 1, current_state.lowest_free_rows[col_ix + 1])
#             if not anchor_row + 3 > current_state.num_rows:
#                 new_lowest_free_rows = current_state.lowest_free_rows.copy()
#                 new_lowest_free_rows[col_ix] = anchor_row + 2
#                 new_lowest_free_rows[col_ix + 1] = anchor_row + 3
#                 new_representation = current_state.representation.copy()
#                 new_representation[anchor_row + 1, col_ix] = 1
#                 new_representation[anchor_row:(anchor_row + 3), col_ix + 1] = 1
#                 new_state = state.State(new_representation,
#                                         new_lowest_free_rows,
#                                         # np.arange(col_ix, col_ix + 2, 1, np.int64),
#                                         np.arange(anchor_row, anchor_row + 2, 1, np.int64),
#                                         np.array([1, 2], dtype=np.int64),
#                                         1,
#                                         self.num_features,
#                                         self.feature_type,
#                                         False)
#
#                                                                                     # current_state.col_transitions_per_col,
#                                             # current_state.row_transitions_per_col,
#                                             # current_state.array_of_rows_with_holes,
#                                             # current_state.holes_per_col,
#                                             # current_state.hole_depths_per_col,
#                                             # current_state.cumulative_wells_per_col)
#                 after_states.append(new_state)
#
#             # Single cell on right
#             anchor_row = np.maximum(current_state.lowest_free_rows[col_ix], current_state.lowest_free_rows[col_ix + 1] - 1)
#             if not anchor_row + 3 > current_state.num_rows:
#                 new_lowest_free_rows = current_state.lowest_free_rows.copy()
#                 new_lowest_free_rows[col_ix] = anchor_row + 3
#                 new_lowest_free_rows[col_ix + 1] = anchor_row + 2
#                 new_representation = current_state.representation.copy()
#                 new_representation[anchor_row:(anchor_row + 3), col_ix] = 1
#                 new_representation[anchor_row + 1, col_ix + 1] = 1
#                 new_state = state.State(new_representation,
#                                         new_lowest_free_rows,
#                                         # np.arange(col_ix, col_ix + 2, 1, np.int64),
#                                         np.arange(anchor_row, anchor_row + 2, 1, np.int64),
#                                         np.array([1, 2], dtype=np.int64),
#                                         1,
#                                         self.num_features,
#                                         self.feature_type,
#                                         False)
#
#                                                                                     # current_state.col_transitions_per_col,
#                                             # current_state.row_transitions_per_col,
#                                             # current_state.array_of_rows_with_holes,
#                                             # current_state.holes_per_col,
#                                             # current_state.hole_depths_per_col,
#                                             # current_state.cumulative_wells_per_col)
#                 after_states.append(new_state)
#
#             if col_ix < self.num_columns - 2:
#                 # Horizontal placements
#         # max_col_index = self.num_columns - 2
#         # for col_ix, free_pos in enumerate(current_state.lowest_free_rows[:max_col_index]):
#             # upside-down T
#                 anchor_row = np.max(current_state.lowest_free_rows[col_ix:(col_ix + 3)])
#                 if not anchor_row + 2 > current_state.num_rows:
#                     new_lowest_free_rows = current_state.lowest_free_rows.copy()
#                     # new_lowest_free_rows[[col_ix, col_ix + 2]] = anchor_row + 1
#                     new_lowest_free_rows[col_ix] = anchor_row + 1
#                     new_lowest_free_rows[col_ix + 2] = anchor_row + 1
#                     new_lowest_free_rows[col_ix + 1] = anchor_row + 2
#                     new_representation = current_state.representation.copy()
#                     new_representation[anchor_row, col_ix:(col_ix + 3)] = 1
#                     new_representation[anchor_row + 1, col_ix + 1] = 1
#                     new_state = state.State(new_representation,
#                                             new_lowest_free_rows,
#                                             # np.arange(col_ix, col_ix + 3, 1, np.int64),
#                                             np.arange(anchor_row, anchor_row + 1, 1, np.int64),
#                                             np.array([3], dtype=np.int64),
#                                             0.5,
#                                             self.num_features,
#                                             self.feature_type,
#                                             False)
#
#                                                                                         # current_state.col_transitions_per_col,
#                                             # current_state.row_transitions_per_col,
#                                             # current_state.array_of_rows_with_holes,
#                                             # current_state.holes_per_col,
#                                             # current_state.hole_depths_per_col,
#                                             # current_state.cumulative_wells_per_col)
#                     after_states.append(new_state)
#
#                 # T
#                 # anchor_row = np.maximum(current_state.lowest_free_rows[col_ix + 1], np.max(current_state.lowest_free_rows[[col_ix, col_ix + 2]]) - 1)
#                 anchor_row = np.maximum(current_state.lowest_free_rows[col_ix + 1],
#                                         np.maximum(current_state.lowest_free_rows[col_ix],
#                                                    current_state.lowest_free_rows[col_ix + 2]) - 1)
#                 if not anchor_row + 2 > current_state.num_rows:
#                     new_lowest_free_rows = current_state.lowest_free_rows.copy()
#                     new_lowest_free_rows[col_ix: (col_ix + 3)] = anchor_row + 2
#                     new_representation = current_state.representation.copy()
#                     new_representation[anchor_row + 1, col_ix:(col_ix + 3)] = 1
#                     new_representation[anchor_row, col_ix + 1] = 1
#                     new_state = state.State(new_representation,
#                                             new_lowest_free_rows,
#                                             # np.arange(col_ix, col_ix + 3, 1, np.int64),
#                                             np.arange(anchor_row, anchor_row + 2, 1, np.int64),
#                                             np.array([1, 3], dtype=np.int64),
#                                             0.5,
#                                             self.num_features,
#                                             self.feature_type,
#                                             False)
#
#                                                                                         # current_state.col_transitions_per_col,
#                                             # current_state.row_transitions_per_col,
#                                             # current_state.array_of_rows_with_holes,
#                                             # current_state.holes_per_col,
#                                             # current_state.hole_depths_per_col,
#                                             # current_state.cumulative_wells_per_col)
#                     after_states.append(new_state)
#         return after_states
#
#
# @jitclass(specT)
# class RCorner:
#     def __init__(self, feature_type, num_features, num_columns):
#         self.feature_type = feature_type
#         self.num_features = num_features
#         self.num_columns = num_columns
#
# #     def __repr__(self):
# #         return """
# # ██ ██ ██
# # ██"""
#
#     def get_after_states(self, current_state):
#         # after_states = List()
#         after_states = []
#         # Vertical placements.
#         max_col_index = self.num_columns - 1
#         for col_ix, free_pos in enumerate(current_state.lowest_free_rows[:max_col_index]):
#             # Top-right corner
#             anchor_row = np.maximum(current_state.lowest_free_rows[col_ix] - 2, current_state.lowest_free_rows[col_ix + 1])
#             if not anchor_row + 3 > current_state.num_rows:
#                 new_lowest_free_rows = current_state.lowest_free_rows.copy()
#                 new_lowest_free_rows[col_ix: (col_ix + 2)] = anchor_row + 3
#                 new_representation = current_state.representation.copy()
#                 new_representation[anchor_row + 2, col_ix] = 1
#                 new_representation[anchor_row:(anchor_row + 3), col_ix + 1] = 1
#                 new_state = state.State(new_representation,
#                                         new_lowest_free_rows,
#                                         # np.arange(col_ix, col_ix + 2, 1, np.int64),
#                                         np.arange(anchor_row, anchor_row + 3, 1, np.int64),
#                                         np.array([1, 1, 2], dtype=np.int64),
#                                         1,
#                                         self.num_features,
#                                         self.feature_type,
#                                         False)
#
#                                                                                     # current_state.col_transitions_per_col,
#                                             # current_state.row_transitions_per_col,
#                                             # current_state.array_of_rows_with_holes,
#                                             # current_state.holes_per_col,
#                                             # current_state.hole_depths_per_col,
#                                             # current_state.cumulative_wells_per_col)
#                 after_states.append(new_state)
#
#             # Bottom-left corner
#             anchor_row = np.max(current_state.lowest_free_rows[col_ix:(col_ix + 2)])
#             if not anchor_row + 3 > current_state.num_rows:
#                 new_lowest_free_rows = current_state.lowest_free_rows.copy()
#                 new_lowest_free_rows[col_ix] = anchor_row + 3
#                 new_lowest_free_rows[col_ix + 1] = anchor_row + 1
#                 new_representation = current_state.representation.copy()
#                 new_representation[anchor_row:(anchor_row + 3), col_ix] = 1
#                 new_representation[anchor_row, col_ix + 1] = 1
#                 new_state = state.State(new_representation,
#                                         new_lowest_free_rows,
#                                         # np.arange(col_ix, col_ix + 2, 1, np.int64),
#                                         np.arange(anchor_row, anchor_row + 1, 1, np.int64),
#                                         np.array([2], dtype=np.int64),
#                                         1,
#                                         self.num_features,
#                                         self.feature_type,
#                                         False)
#
#                                                                                     # current_state.col_transitions_per_col,
#                                             # current_state.row_transitions_per_col,
#                                             # current_state.array_of_rows_with_holes,
#                                             # current_state.holes_per_col,
#                                             # current_state.hole_depths_per_col,
#                                             # current_state.cumulative_wells_per_col)
#                 after_states.append(new_state)
#
#             if col_ix < self.num_columns - 2:
#                 # Horizontal placements
#         # max_col_index = self.num_columns - 2
#         # for col_ix, free_pos in enumerate(current_state.lowest_free_rows[:max_col_index]):
#                 # Bottom-right corner
#                 anchor_row = np.max(current_state.lowest_free_rows[col_ix:(col_ix + 3)])
#                 if not anchor_row + 2 > current_state.num_rows:
#                     new_lowest_free_rows = current_state.lowest_free_rows.copy()
#                     new_lowest_free_rows[col_ix:(col_ix + 2)] = anchor_row + 1
#                     new_lowest_free_rows[col_ix + 2] = anchor_row + 2
#                     new_representation = current_state.representation.copy()
#                     new_representation[anchor_row, col_ix:(col_ix + 3)] = 1
#                     new_representation[anchor_row + 1, col_ix + 2] = 1
#                     new_state = state.State(new_representation,
#                                             new_lowest_free_rows,
#                                             # np.arange(col_ix, col_ix + 3, 1, np.int64),
#                                             np.arange(anchor_row, anchor_row + 1, 1, np.int64),
#                                             np.array([3], dtype=np.int64),
#                                             0.5,
#                                             self.num_features,
#                                             self.feature_type,
#                                             False)
#
#                                                                                         # current_state.col_transitions_per_col,
#                                             # current_state.row_transitions_per_col,
#                                             # current_state.array_of_rows_with_holes,
#                                             # current_state.holes_per_col,
#                                             # current_state.hole_depths_per_col,
#                                             # current_state.cumulative_wells_per_col)
#                     after_states.append(new_state)
#
#                 # Top-left corner
#                 anchor_row = np.maximum(current_state.lowest_free_rows[col_ix], np.max(current_state.lowest_free_rows[(col_ix + 1):(col_ix + 3)]) - 1)
#                 if not anchor_row + 2 > current_state.num_rows:
#                     new_lowest_free_rows = current_state.lowest_free_rows.copy()
#                     new_lowest_free_rows[col_ix: (col_ix + 3)] = anchor_row + 2
#                     new_representation = current_state.representation.copy()
#                     new_representation[anchor_row + 1, col_ix:(col_ix + 3)] = 1
#                     new_representation[anchor_row, col_ix] = 1
#                     new_state = state.State(new_representation,
#                                             new_lowest_free_rows,
#                                             # np.arange(col_ix, col_ix + 3, 1, np.int64),
#                                             np.arange(anchor_row, anchor_row + 2, 1, np.int64),
#                                             np.array([1, 3], dtype=np.int64),
#                                             0.5,
#                                             self.num_features,
#                                             self.feature_type,
#                                             False)
#
#                                                                                         # current_state.col_transitions_per_col,
#                                             # current_state.row_transitions_per_col,
#                                             # current_state.array_of_rows_with_holes,
#                                             # current_state.holes_per_col,
#                                             # current_state.hole_depths_per_col,
#                                             # current_state.cumulative_wells_per_col)
#                     after_states.append(new_state)
#         return after_states
#
#
# @jitclass(specT)
# class LCorner:
#     def __init__(self, feature_type, num_features, num_columns):
#         self.feature_type = feature_type
#         self.num_features = num_features
#         self.num_columns = num_columns
#
# #     def __repr__(self):
# #         return """
# # ██ ██ ██
# #       ██"""
#
#     def get_after_states(self, current_state):
#         # after_states = List()
#         after_states = []
#         # Vertical placements. 'height' becomes 'width' :)
#         max_col_index = self.num_columns - 1
#         for col_ix, free_pos in enumerate(current_state.lowest_free_rows[:max_col_index]):
#             # Top-left corner
#             anchor_row = np.maximum(current_state.lowest_free_rows[col_ix], current_state.lowest_free_rows[col_ix + 1] - 2)
#             if not anchor_row + 3 > current_state.num_rows:
#                 new_lowest_free_rows = current_state.lowest_free_rows.copy()
#                 new_lowest_free_rows[col_ix: (col_ix + 2)] = anchor_row + 3
#                 new_representation = current_state.representation.copy()
#                 new_representation[anchor_row + 2, col_ix + 1] = 1
#                 new_representation[anchor_row:(anchor_row + 3), col_ix] = 1
#                 new_state = state.State(new_representation,
#                                         new_lowest_free_rows,
#                                         # np.arange(col_ix, col_ix + 2, 1, np.int64),
#                                         np.arange(anchor_row, anchor_row + 3, 1, np.int64),
#                                         np.array([1, 1, 2], dtype=np.int64),
#                                         1,
#                                         self.num_features,
#                                         self.feature_type,
#                                         False)
#
#                                                                                     # current_state.col_transitions_per_col,
#                                             # current_state.row_transitions_per_col,
#                                             # current_state.array_of_rows_with_holes,
#                                             # current_state.holes_per_col,
#                                             # current_state.hole_depths_per_col,
#                                             # current_state.cumulative_wells_per_col)
#                 after_states.append(new_state)
#
#             # Bottom-right corner
#             anchor_row = np.max(current_state.lowest_free_rows[col_ix:(col_ix + 2)])
#             if not anchor_row + 3 > current_state.num_rows:
#                 new_lowest_free_rows = current_state.lowest_free_rows.copy()
#                 new_lowest_free_rows[col_ix + 1] = anchor_row + 3
#                 new_lowest_free_rows[col_ix] = anchor_row + 1
#                 new_representation = current_state.representation.copy()
#                 new_representation[anchor_row:(anchor_row + 3), col_ix + 1] = 1
#                 new_representation[anchor_row, col_ix] = 1
#                 new_state = state.State(new_representation,
#                                         new_lowest_free_rows,
#                                         # np.arange(col_ix, col_ix + 2, 1, np.int64),
#                                         np.arange(anchor_row, anchor_row + 1, 1, np.int64),
#                                         np.array([2], dtype=np.int64),
#                                         1,
#                                         self.num_features,
#                                         self.feature_type,
#                                         False)
#
#                                                                                     # current_state.col_transitions_per_col,
#                                             # current_state.row_transitions_per_col,
#                                             # current_state.array_of_rows_with_holes,
#                                             # current_state.holes_per_col,
#                                             # current_state.hole_depths_per_col,
#                                             # current_state.cumulative_wells_per_col)
#                 after_states.append(new_state)
#
#
#             if col_ix < self.num_columns - 2:
#
#         # max_col_index = self.num_columns - 2
#         # for col_ix, free_pos in enumerate(current_state.lowest_free_rows[:max_col_index]):
#                 # Bottom-left corner (= 'hole' in top-right corner)
#                 anchor_row = np.max(current_state.lowest_free_rows[col_ix:(col_ix + 3)])
#                 if not anchor_row + 2 > current_state.num_rows:
#                     new_lowest_free_rows = current_state.lowest_free_rows.copy()
#                     new_lowest_free_rows[col_ix] = anchor_row + 2
#                     new_lowest_free_rows[(col_ix + 1):(col_ix + 3)] = anchor_row + 1
#                     new_representation = current_state.representation.copy()
#                     new_representation[anchor_row, col_ix:(col_ix + 3)] = 1
#                     new_representation[anchor_row + 1, col_ix] = 1
#                     new_state = state.State(new_representation,
#                                             new_lowest_free_rows,
#                                             # np.arange(col_ix, col_ix + 3, 1, np.int64),
#                                             np.arange(anchor_row, anchor_row + 1, 1, np.int64),
#                                             np.array([3], dtype=np.int64),
#                                             0.5,
#                                             self.num_features,
#                                             self.feature_type,
#                                             False)
#                                                                                         # current_state.col_transitions_per_col,
#                                             # current_state.row_transitions_per_col,
#                                             # current_state.array_of_rows_with_holes,
#                                             # current_state.holes_per_col,
#                                             # current_state.hole_depths_per_col,
#                                             # current_state.cumulative_wells_per_col)
#                     after_states.append(new_state)
#
#                 # Top-right corner
#                 anchor_row = np.maximum(np.max(current_state.lowest_free_rows[col_ix:(col_ix + 2)]) - 1, current_state.lowest_free_rows[col_ix + 2])
#                 if not anchor_row + 2 > current_state.num_rows:
#                     new_lowest_free_rows = current_state.lowest_free_rows.copy()
#                     new_lowest_free_rows[col_ix: (col_ix + 3)] = anchor_row + 2
#                     new_representation = current_state.representation.copy()
#                     new_representation[anchor_row + 1, col_ix:(col_ix + 3)] = 1
#                     new_representation[anchor_row, col_ix + 2] = 1
#                     new_state = state.State(new_representation,
#                                             new_lowest_free_rows,
#                                             # np.arange(col_ix, col_ix + 3, 1, np.int64),
#                                             np.arange(anchor_row, anchor_row + 2, 1, np.int64),
#                                             np.array([1, 3], dtype=np.int64),
#                                             0.5,
#                                             self.num_features,
#                                             self.feature_type,
#                                             False)
#
#                                                                                         # current_state.col_transitions_per_col,
#                                             # current_state.row_transitions_per_col,
#                                             # current_state.array_of_rows_with_holes,
#                                             # current_state.holes_per_col,
#                                             # current_state.hole_depths_per_col,
#                                             # current_state.cumulative_wells_per_col)
#                     after_states.append(new_state)
#         return after_states
#
# #
# # @njit(cache=False)
# # def numba_any(arr):
# #     found = False
# #     i = 0
# #     arr_len = len(arr)
# #     while not found and i < arr_len:
# #         if arr[i]:
# #             found = True
# #         i += 1
# #     return found
#
#
#
# # class Tetromino:
# #     def __init__(self, feature_type, num_features, num_columns):
# #         self.feature_type = feature_type
# #         self.num_features = num_features
# #         self.num_columns = num_columns
#
#
#
#
# # class TetrominoSampler:
# #     def __init__(self, tetrominos):
# #         self.tetrominos = tetrominos
# #         self.current_batch = np.random.permutation(len(self.tetrominos))
# #
# #     def next_tetromino(self):
# #         if len(self.current_batch) == 0:
# #             self.current_batch = np.random.permutation(len(self.tetrominos))
# #         tetromino = self.tetrominos[self.current_batch[0]]
# #         self.current_batch = np.delete(self.current_batch, 0)
# #         return tetromino
#
#
# # @njit
# # def numba_any_break(arr):
# #     found = False
# #     for i in arr:
# #         if i:
# #             found = True
# #             break
# #     return found
#
#
#
