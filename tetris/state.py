import numpy as np
import numba
from numba import njit, jitclass, float64, int8, bool_, int64
# import torch

specTerm = [
    ('terminal_state', bool_),
]

@jitclass(specTerm)
class TerminalState:
    def __init__(self):
        self.terminal_state = True


spec = [
    ('representation', bool_[:, :]),
    ('lowest_free_rows', int8[:]),
    # ('anchor_col', int8),
    ('changed_lines', int8[:]),
    ('pieces_per_changed_row', int8[:]),
    ('landing_height_bonus', float64),
    ('num_features', int8),
    ('feature_type', numba.types.string),
    ('num_rows', int8),
    ('num_columns', int64),
    ('n_cleared_lines', int8),
    # ('changed_cols', int8[:]),
    ('anchor_row', int8),
    ('cleared_rows_relative_to_anchor', bool_[:]),
    ('features_are_calculated', bool_),
    ('features', float64[:]),
    ('terminal_state', bool_)  #,
    # ('reward', int8),
    # ('value_estimate', float64),
    # ('col_transitions_per_col', int8[:]),
    # ('row_transitions_per_col', int8[:]),
    # ('holes_per_col', int8[:]),
    # ('hole_depths_per_col', int8[:]),
    # ('cumulative_wells_per_col', int8[:]),
    # ('array_of_rows_with_holes', int8[:])
]

# FOR SET: numba.types.containers.Set

@jitclass(spec)
class State(object):
    def __init__(self,
                 representation,
                 lowest_free_rows,
                 # changed_cols, #=np.array([0], dtype=np.int8),
                 changed_lines, #=np.array([0], dtype=np.int8),
                 pieces_per_changed_row, #=np.array([0], dtype=np.int8),
                 landing_height_bonus, # =0.0,
                 num_features, #=8,
                 feature_type, #="bcts",
                 terminal_state
                 # col_transitions_per_col, #=np.array([0], dtype=np.int8),
                 # row_transitions_per_col, #=np.array([0], dtype=np.int8),
                 # array_of_rows_with_holes, #=np.array([100], dtype=np.int8),
                 # holes_per_col, #=np.array([0], dtype=np.int8),
                 # hole_depths_per_col, #=np.array([0], dtype=np.int8),
                 # cumulative_wells_per_col #=np.array([0], dtype=np.int8)):  # , features=np.zeros(1, dtype=np.float64)
                 ):
        self.terminal_state = terminal_state
        if not terminal_state:
            self.representation = representation
            self.lowest_free_rows = lowest_free_rows
            self.num_rows, self.num_columns = representation.shape
            self.pieces_per_changed_row = pieces_per_changed_row
            self.landing_height_bonus = landing_height_bonus
            self.num_features = num_features
            self.feature_type = feature_type  # "bcts"
            self.n_cleared_lines = 0
            self.anchor_row = changed_lines[0]
            self.cleared_rows_relative_to_anchor = self.clear_lines(changed_lines)
            # # TODO: REMOVE FOR SPEED
            # assert np.all(calc_lowest_free_rows(self.representation) == self.lowest_free_rows)
            self.features_are_calculated = False
            self.features = np.zeros(self.num_features, dtype=np.float64)

        # self.reward = 0 if self.terminal_state else self.n_cleared_lines
        # self.value_estimate = 0.0
        # self.anchor_col = changed_cols[0]
        # self.col_transitions_per_col = col_transitions_per_col.copy()
        # self.row_transitions_per_col = row_transitions_per_col.copy()
        # self.array_of_rows_with_holes = array_of_rows_with_holes.copy()
        # self.holes_per_col = holes_per_col.copy()
        # self.hole_depths_per_col = hole_depths_per_col.copy()
        # self.cumulative_wells_per_col = cumulative_wells_per_col.copy()

        # self.col_transitions_per_col = np.zeros(self.num_columns, dtype=np.int8)
        # self.row_transitions_per_col = np.zeros(self.num_columns+1, dtype=np.int8)
        # self.array_of_rows_with_holes = np.array([100], dtype=np.int8)
        # self.holes_per_col = np.zeros(self.num_columns, dtype=np.int8)
        # self.hole_depths_per_col = np.zeros(self.num_columns, dtype=np.int8)
        # self.cumulative_wells_per_col = np.zeros(self.num_columns, dtype=np.int8)

    # def __repr__(self):
    #     return self.print_board_to_string()
    #
    # def print_board_to_string(self):
    #     string = "\n"
    #     for row_ix in range(self.num_rows):
    #         # Start from top
    #         row_ix = self.num_rows - row_ix - 1
    #         string += "|"
    #         for col_ix in range(self.num_columns):
    #             if self.representation[row_ix, col_ix]:
    #                 string += "██"
    #             else:
    #                 string += "  "
    #         string += "|\n"
    #     return string

    def get_features(self, direct_by, addRBF=False):  #, order_by=None, standardize_by=None, addRBF=False
        if not self.features_are_calculated:
            if self.feature_type == "bcts":
                # if self.n_cleared_lines > 0:
                self.calc_bcts_features()
                # else:
                #     # self.update_bcts_features()
                #     # UPDATED_FEATURES = self.features.copy()
                #     # self.calc_bcts_features()
                #     # if np.any(UPDATED_FEATURES != self.features):
                #     #     print("BLA")
                #     self.calc_bcts_features()
                # self.features_are_calculated = True
            else:
                raise ValueError("Feature type must be either bcts or standardized_bcts or simple or super_simple")
        # TODO: check whether copy is needed here.
        out = self.features * direct_by # .copy()
        # features = features
        # if order_by is not None:
        #     features = features[order_by]
        # if standardize_by is not None:
        #     features = features / standardize_by
        if addRBF:
            out = np.concatenate((
                out,
                np.exp(-(np.mean(self.lowest_free_rows) - np.arange(5) * self.num_rows / 4) ** 2 / (2 * (self.num_rows / 5) ** 2))
                ))
        return out

    # TODO: Implement order / directions...
    # def get_features_with_intercept(self):
    #     if self.features is None:
    #         self.calc_feature_values()
    #     return np.insert(self.features, obj=0, values=1.)

    # def clear_lines(self, changed_lines):
    #     is_full, self.n_cleared_lines, self.representation, self.lowest_free_rows = \
    #         clear_lines_jitted(changed_lines, self.representation,
    #                            self.lowest_free_rows, self.num_columns)
    #     return is_full

    def clear_lines(self, changed_lines):
        num_columns = self.num_columns
        row_sums = np.sum(self.representation[changed_lines, :], axis=1)
        is_full = (row_sums == num_columns)
        full_lines = np.where(is_full)[0]
        n_cleared_lines = len(full_lines)
        if n_cleared_lines > 0:
            # print(self)
            representation = self.representation
            lowest_free_rows = self.lowest_free_rows
            lines_to_clear = changed_lines[full_lines].astype(np.int8)
            mask_keep = np.ones(len(representation), dtype=np.bool_)
            mask_keep[lines_to_clear] = False
            new_cols = np.zeros((n_cleared_lines, num_columns), dtype=np.bool_)
            representation = np.vstack((representation[mask_keep], new_cols))
            for col_ix in range(num_columns):  # col_ix = 0
                old_lowest_free_row = lowest_free_rows[col_ix]
                if old_lowest_free_row > lines_to_clear[-1] + 1:
                    lowest_free_rows[col_ix] -= n_cleared_lines
                else:
                    lowest = 0
                    for row_ix in range(old_lowest_free_row - n_cleared_lines - 1, -1, -1):
                        if representation[row_ix, col_ix] == 1:
                            lowest = row_ix + 1
                            break
                    lowest_free_rows[col_ix] = lowest
            self.lowest_free_rows = lowest_free_rows
            self.representation = representation

        self.n_cleared_lines = n_cleared_lines
        return is_full  # , n_cleared_lines, representation, lowest_free_rows
    # def update_bcts_features(self, old_feature_values, old_rows_with_holes):
    #     pass

    # TODO: Optimization ideas: representation to bools
    # TODO:                     only count hole depth first time when an actual hole is found
    # TODO:                     minimize calls to self (define variables in method) or completely outsource it!?
    # TODO:                     only create empty vectors of size "relevant cols".

    def calc_bcts_features(self):
        rows_with_holes_set = {100}
        representation = self.representation
        num_rows, num_columns = representation.shape
        lowest_free_rows = self.lowest_free_rows
        # col_transitions_per_col = np.zeros(num_columns, dtype=np.int8)
        # row_transitions_per_col = np.zeros(num_columns + 1, dtype=np.int8)
        # holes_per_col = np.zeros(num_columns, dtype=np.int8)
        # hole_depths_per_col = np.zeros(num_columns, dtype=np.int8)
        # cumulative_wells_per_col = np.zeros(num_columns, dtype=np.int8)
        col_transitions = 0
        row_transitions = 0
        holes = 0
        hole_depths = 0
        cumulative_wells = 0
        # row_transitions = 0
        for col_ix, lowest_free_row in enumerate(lowest_free_rows):
            # There is at least one column_transition from the highest full cell (or the bottom which is assumed to be full) to "the top".
            col_transitions += 1
            if col_ix == 0:
                local_well_streak = 0
                if lowest_free_row > 0:
                    col = representation[:lowest_free_row, col_ix]
                    cell_below = 1

                    # Needed for hole_depth
                    # TODO: Optimize... only count the first time when an actual hole is found
                    number_of_full_cells_above = numba_sum_int(col)

                    for row_ix, cell in enumerate(col):
                        if cell == 0:
                            # Holes
                            holes += 1
                            rows_with_holes_set.add(row_ix)
                            hole_depths += number_of_full_cells_above

                            # Column transitions
                            if cell_below:
                                col_transitions += 1

                            # Row transitions and wells
                            # Because col_ix == 0, all left_cells are 1
                            # row_transitions += 1
                            row_transitions += 1
                            if representation[row_ix, col_ix + 1]:  # if cell to the right is full
                                local_well_streak += 1
                                cumulative_wells += local_well_streak
                            else:
                                local_well_streak = 0

                        else:  # cell is 1!
                            local_well_streak = 0

                            # Keep track of full cells above for hole_depth-feature
                            number_of_full_cells_above -= 1

                            # Column transitions
                            if not cell_below:
                                col_transitions += 1

                        # Define 'cell_below' for next (higher!) cell.
                        cell_below = cell

                # Check wells until lowest_free_row_right
                # Check transitions until lowest_free_row_left
                max_well_possibility = lowest_free_rows[col_ix + 1]
                if max_well_possibility > lowest_free_row:
                    for row_ix in range(lowest_free_row, max_well_possibility):
                        if representation[row_ix, col_ix + 1]:  # if cell to the right is full
                            local_well_streak += 1
                            cumulative_wells += local_well_streak
                        else:
                            local_well_streak = 0
                # # Add row transitions for each empty cell above lowest_free_row
                row_transitions += (num_rows - lowest_free_row)

            elif col_ix == num_columns - 1:
                local_well_streak = 0
                if lowest_free_row > 0:
                    col = representation[:lowest_free_row, col_ix]
                    cell_below = 1

                    # Needed for hole_depth
                    number_of_full_cells_above = numba_sum_int(col)

                    for row_ix, cell in enumerate(col):
                        if cell == 0:
                            # Holes
                            holes += 1
                            rows_with_holes_set.add(row_ix)
                            hole_depths += number_of_full_cells_above

                            # Column transitions
                            if cell_below:
                                col_transitions += 1

                            # Wells and row transitions
                            # Because this is the last column (the right border is "full") and cell == 0:
                            row_transitions += 1
                            if representation[row_ix, col_ix - 1]:  # if cell to the left is full
                                row_transitions += 1
                                local_well_streak += 1
                                cumulative_wells += local_well_streak
                            else:
                                local_well_streak = 0

                        else:  # cell is 1!
                            local_well_streak = 0

                            # Keep track of full cells above for hole_depth-feature
                            number_of_full_cells_above -= 1

                            # Column transitions
                            if not cell_below:
                                col_transitions += 1

                            # Row transitions
                            cell_left = representation[row_ix, col_ix - 1]
                            if not cell_left:
                                row_transitions += 1

                        # Define 'cell_below' for next (higher!) cell.
                        cell_below = cell

                # Check wells until minimum(lowest_free_row_left, lowest_free_row_right)
                # Check transitions until lowest_free_row_left
                max_well_possibility = lowest_free_rows[col_ix - 1]
                if max_well_possibility > lowest_free_row:
                    for row_ix in range(lowest_free_row, max_well_possibility):
                        if representation[row_ix, col_ix - 1]:  # if cell to the left is full
                            row_transitions += 1
                            local_well_streak += 1
                            cumulative_wells += local_well_streak
                        else:
                            local_well_streak = 0
                # # Add row transitions from last column to border
                row_transitions += (num_rows - lowest_free_row)
            else:
                local_well_streak = 0
                if lowest_free_row > 0:
                    col = representation[:lowest_free_row, col_ix]
                    cell_below = 1

                    # Needed for hole_depth
                    number_of_full_cells_above = numba_sum_int(col)

                    for row_ix, cell in enumerate(col):
                        if cell == 0:
                            # Holes
                            holes += 1
                            rows_with_holes_set.add(row_ix)
                            hole_depths += number_of_full_cells_above

                            # Column transitions
                            if cell_below:
                                col_transitions += 1

                            # Wells and row transitions
                            cell_left = representation[row_ix, col_ix - 1]
                            if cell_left:
                                row_transitions += 1
                                cell_right = representation[row_ix, col_ix + 1]
                                if cell_right:
                                    local_well_streak += 1
                                    cumulative_wells += local_well_streak
                                else:
                                    local_well_streak = 0
                            else:
                                local_well_streak = 0

                        else:  # cell is 1!
                            local_well_streak = 0
                            # Keep track of full cells above for hole_depth-feature
                            number_of_full_cells_above -= 1

                            # Column transitions
                            if not cell_below:
                                col_transitions += 1

                            # Row transitions
                            cell_left = representation[row_ix, col_ix - 1]
                            if not cell_left:
                                row_transitions += 1

                        # Define 'cell_below' for next (higher!) cell.
                        cell_below = cell
                # Check wells until minimum(lowest_free_row_left, lowest_free_row_right)
                # Check transitions until lowest_free_row_left
                lowest_free_row_left = lowest_free_rows[col_ix - 1]
                lowest_free_row_right = lowest_free_rows[col_ix + 1]
                max_well_possibility = np.minimum(lowest_free_row_left, lowest_free_row_right)

                # Weird case distinction because max_well_possibility always "includes" lowest_free_row_left
                #  but lowest_free_row_left can be higher than max_well_possibility. Don't want to double count.
                if max_well_possibility > lowest_free_row:
                    for row_ix in range(lowest_free_row, max_well_possibility):
                        cell_left = representation[row_ix, col_ix - 1]
                        if cell_left:
                            row_transitions += 1
                            cell_right = representation[row_ix, col_ix + 1]
                            if cell_right:
                                local_well_streak += 1
                                cumulative_wells += local_well_streak
                            else:
                                local_well_streak = 0
                        else:
                            local_well_streak = 0
                    if lowest_free_row_left > max_well_possibility:
                        for row_ix in range(max_well_possibility, lowest_free_row_left):
                            cell_left = representation[row_ix, col_ix - 1]
                            if cell_left:
                                row_transitions += 1
                elif lowest_free_row_left > lowest_free_row:
                    for row_ix in range(lowest_free_row, lowest_free_row_left):
                        cell_left = representation[row_ix, col_ix - 1]
                        if cell_left:
                            row_transitions += 1

        rows_with_holes_set.remove(100)
        rows_with_holes = len(rows_with_holes_set)
        eroded_pieces = numba_sum_int(self.cleared_rows_relative_to_anchor * self.pieces_per_changed_row)
        # n_cleared_lines = numba_sum_int(self.cleared_rows_relative_to_anchor)
        eroded_piece_cells = eroded_pieces * self.n_cleared_lines
        landing_height = self.anchor_row + self.landing_height_bonus
        self.features = np.array([rows_with_holes, col_transitions, holes, landing_height,
                                  cumulative_wells, row_transitions, eroded_piece_cells, hole_depths])


    # def update_bcts_features(self):
    #     # Can only add holes... not remove them
    #     rows_with_holes_set = {100}
    #     representation = self.representation
    #     num_rows, num_columns = representation.shape
    #     lowest_free_rows = self.lowest_free_rows
    #     min_relevant_col = np.maximum(self.changed_cols[0]-1, 0)
    #     max_relevant_col = np.minimum(self.changed_cols[-1]+1, num_columns-1)
    #     relevant_cols = np.arange(min_relevant_col, max_relevant_col + 1)
    #     num_relevant_cols = len(relevant_cols)
    #     # relevant_lowest_free_rows = lowest_free_rows
    #     col_transitions_per_col = np.zeros(num_columns, dtype=np.int8)
    #     row_transitions_per_col = np.zeros(num_columns + 1, dtype=np.int8)
    #     holes_per_col = np.zeros(num_columns, dtype=np.int8)
    #     hole_depths_per_col = np.zeros(num_columns, dtype=np.int8)
    #     cumulative_wells_per_col = np.zeros(num_columns, dtype=np.int8)
    #     for col_ix in relevant_cols:
    #         lowest_free_row = lowest_free_rows[col_ix]
    #         # There is at least one column_transition from the highest full cell (or the bottom which is assumed to be full) to "the top".
    #         col_transitions_per_col[col_ix] += 1
    #         if col_ix == 0:
    #             local_well_streak = 0
    #             if lowest_free_row > 0:
    #                 col = representation[:lowest_free_row, col_ix]
    #                 cell_below = 1
    #
    #                 # Needed for hole_depth
    #                 # TODO: Optimize... only count the first time when an actual hole is found
    #                 number_of_full_cells_above = numba_sum_int(col)
    #
    #                 for row_ix, cell in enumerate(col):
    #                     if cell == 0:
    #                         # Holes
    #                         holes_per_col[col_ix] += 1
    #                         rows_with_holes_set.add(row_ix)
    #                         hole_depths_per_col[col_ix] += number_of_full_cells_above
    #
    #                         # Column transitions
    #
    #                         if cell_below:
    #                             col_transitions_per_col[col_ix] += 1
    #
    #                         # Row transitions and wells
    #                         # Because col_ix == 0, all left_cells are 1
    #                         row_transitions_per_col[col_ix] += 1
    #                         if representation[row_ix, col_ix + 1]:  # if cell to the right is full
    #                             local_well_streak += 1
    #                             cumulative_wells_per_col[col_ix] += local_well_streak
    #                         else:
    #                             local_well_streak = 0
    #
    #                     else:  # cell is 1!
    #                         local_well_streak = 0
    #
    #                         # Keep track of full cells above for hole_depth-feature
    #                         number_of_full_cells_above -= 1
    #
    #                         # Column transitions
    #                         if not cell_below:
    #                             col_transitions_per_col[col_ix] += 1
    #
    #                     # Define 'cell_below' for next (higher!) cell.
    #                     cell_below = cell
    #
    #             # Check wells until lowest_free_row_right
    #             # Check transitions until lowest_free_row_left
    #             max_well_possibility = lowest_free_rows[col_ix + 1]
    #             if max_well_possibility > lowest_free_row:
    #                 for row_ix in range(lowest_free_row, max_well_possibility):
    #                     if representation[row_ix, col_ix + 1]:  # if cell to the right is full
    #                         local_well_streak += 1
    #                         cumulative_wells_per_col[col_ix] += local_well_streak
    #                     else:
    #                         local_well_streak = 0
    #             # # Add row transitions for each empty cell above lowest_free_row
    #             row_transitions_per_col[col_ix] += (num_rows - lowest_free_row)
    #
    #         elif col_ix == num_columns - 1:
    #             local_well_streak = 0
    #             if lowest_free_row > 0:
    #                 col = representation[:lowest_free_row, col_ix]
    #                 cell_below = 1
    #
    #                 # Needed for hole_depth
    #                 number_of_full_cells_above = numba_sum_int(col)
    #
    #                 for row_ix, cell in enumerate(col):
    #                     if cell == 0:
    #                         # Holes
    #                         holes_per_col[col_ix] += 1
    #                         rows_with_holes_set.add(row_ix)
    #                         hole_depths_per_col[col_ix] += number_of_full_cells_above
    #
    #                         # Column transitions
    #                         if cell_below:
    #                             col_transitions_per_col[col_ix] += 1
    #
    #                         # Wells and row transitions
    #                         # Because this is the last column (the right border is "full") and cell == 0:
    #                         row_transitions_per_col[col_ix + 1] += 1
    #                         if representation[row_ix, col_ix - 1]:  # if cell to the left is full
    #                             row_transitions_per_col[col_ix] += 1
    #                             local_well_streak += 1
    #                             cumulative_wells_per_col[col_ix] += local_well_streak
    #                         else:
    #                             local_well_streak = 0
    #
    #                     else:  # cell is 1!
    #                         local_well_streak = 0
    #
    #                         # Keep track of full cells above for hole_depth-feature
    #                         number_of_full_cells_above -= 1
    #
    #                         # Column transitions
    #                         if not cell_below:
    #                             col_transitions_per_col[col_ix] += 1
    #
    #                         # Row transitions
    #                         # # Important: delete in col_ix + 1 because it starts from previous state
    #                         # row_transitions_per_col[col_ix + 1] -= 1
    #
    #                         cell_left = representation[row_ix, col_ix - 1]
    #                         if not cell_left:
    #                             row_transitions_per_col[col_ix] += 1
    #
    #                     # Define 'cell_below' for next (higher!) cell.
    #                     cell_below = cell
    #
    #             # Check wells until minimum(lowest_free_row_left, lowest_free_row_right)
    #             # Check transitions until lowest_free_row_left
    #             max_well_possibility = lowest_free_rows[col_ix - 1]
    #             if max_well_possibility > lowest_free_row:
    #                 for row_ix in range(lowest_free_row, max_well_possibility):
    #                     if representation[row_ix, col_ix - 1]:  # if cell to the left is full
    #                         row_transitions_per_col[col_ix] += 1
    #                         local_well_streak += 1
    #                         cumulative_wells_per_col[col_ix] += local_well_streak
    #                     else:
    #                         local_well_streak = 0
    #             # # Add row transitions from last column to border
    #             row_transitions_per_col[col_ix + 1] += (num_rows - lowest_free_row)
    #         else:
    #             local_well_streak = 0
    #             if lowest_free_row > 0:
    #                 col = representation[:lowest_free_row, col_ix]
    #                 cell_below = 1
    #
    #                 # Needed for hole_depth
    #                 number_of_full_cells_above = numba_sum_int(col)
    #
    #                 for row_ix, cell in enumerate(col):
    #                     if cell == 0:
    #                         # Holes
    #                         holes_per_col[col_ix] += 1
    #                         rows_with_holes_set.add(row_ix)
    #                         hole_depths_per_col[col_ix] += number_of_full_cells_above
    #
    #                         # Column transitions
    #                         if cell_below:
    #                             col_transitions_per_col[col_ix] += 1
    #
    #                         # Wells and row transitions
    #                         cell_left = representation[row_ix, col_ix - 1]
    #                         if cell_left:
    #                             row_transitions_per_col[col_ix] += 1
    #                             cell_right = representation[row_ix, col_ix + 1]
    #                             if cell_right:
    #                                 local_well_streak += 1
    #                                 cumulative_wells_per_col[col_ix] += local_well_streak
    #                             else:
    #                                 local_well_streak = 0
    #                         else:
    #                             local_well_streak = 0
    #
    #                     else:  # cell is 1!
    #                         local_well_streak = 0
    #                         # Keep track of full cells above for hole_depth-feature
    #                         number_of_full_cells_above -= 1
    #
    #                         # Column transitions
    #                         if not cell_below:
    #                             col_transitions_per_col[col_ix] += 1
    #
    #                         # Row transitions
    #                         cell_left = representation[row_ix, col_ix - 1]
    #                         if not cell_left:
    #                             row_transitions_per_col[col_ix] += 1
    #
    #                     # Define 'cell_below' for next (higher!) cell.
    #                     cell_below = cell
    #             # Check wells until minimum(lowest_free_row_left, lowest_free_row_right)
    #             # Check transitions until lowest_free_row_left
    #             lowest_free_row_left = lowest_free_rows[col_ix - 1]
    #             lowest_free_row_right = lowest_free_rows[col_ix + 1]
    #             max_well_possibility = np.minimum(lowest_free_row_left, lowest_free_row_right)
    #
    #             # Weird case distinction because max_well_possibility always "includes" lowest_free_row_left
    #             #  but lowest_free_row_left can be higher than max_well_possibility. Don't want to double count.
    #             if max_well_possibility > lowest_free_row:
    #                 for row_ix in range(lowest_free_row, max_well_possibility):
    #                     cell_left = representation[row_ix, col_ix - 1]
    #                     if cell_left:
    #                         row_transitions_per_col[col_ix] += 1
    #                         cell_right = representation[row_ix, col_ix + 1]
    #                         if cell_right:
    #                             local_well_streak += 1
    #                             cumulative_wells_per_col[col_ix] += local_well_streak
    #                         else:
    #                             local_well_streak = 0
    #                     else:
    #                         local_well_streak = 0
    #                 if lowest_free_row_left > max_well_possibility:
    #                     for row_ix in range(max_well_possibility, lowest_free_row_left):
    #                         cell_left = representation[row_ix, col_ix - 1]
    #                         if cell_left:
    #                             row_transitions_per_col[col_ix] += 1
    #             elif lowest_free_row_left > lowest_free_row:
    #                 for row_ix in range(lowest_free_row, lowest_free_row_left):
    #                     cell_left = representation[row_ix, col_ix - 1]
    #                     if cell_left:
    #                         row_transitions_per_col[col_ix] += 1
    #
    #     rows_with_holes_set.remove(100)
    #     rows_with_holes_set = rows_with_holes_set.union(set(self.array_of_rows_with_holes))
    #     self.array_of_rows_with_holes = np.array(list(rows_with_holes_set), dtype=np.int8)
    #     rows_with_holes = len(rows_with_holes_set)
    #     self.col_transitions_per_col[relevant_cols] = col_transitions_per_col[relevant_cols]
    #     column_transitions = np.sum(self.col_transitions_per_col)
    #     self.holes_per_col[relevant_cols] = holes_per_col[relevant_cols]
    #     holes = np.sum(self.holes_per_col)
    #     self.hole_depths_per_col[relevant_cols] = hole_depths_per_col[relevant_cols]
    #     hole_depth = np.sum(self.hole_depths_per_col)
    #     self.cumulative_wells_per_col[relevant_cols] = cumulative_wells_per_col[relevant_cols]
    #     cumulative_wells = np.sum(self.cumulative_wells_per_col)
    #     self.row_transitions_per_col[relevant_cols] = row_transitions_per_col[relevant_cols]
    #     if max_relevant_col == self.num_columns - 1:
    #         self.row_transitions_per_col[self.num_columns] = row_transitions_per_col[self.num_columns]
    #     row_transitions = np.sum(self.row_transitions_per_col)
    #     eroded_piece_cells = 0
    #     landing_height = self.anchor_row + self.landing_height_bonus
    #     self.features = np.array([rows_with_holes, column_transitions, holes, landing_height,
    #                               cumulative_wells, row_transitions, eroded_piece_cells,
    #                               hole_depth])

    # def calc_bcts_features_per_col(self):
    #     rows_with_holes_set = {100}
    #     representation = self.representation
    #     num_rows, num_columns = representation.shape
    #     lowest_free_rows = self.lowest_free_rows
    #     col_transitions_per_col = np.zeros(num_columns, dtype=np.int8)
    #     row_transitions_per_col = np.zeros(num_columns + 1, dtype=np.int8)
    #     holes_per_col = np.zeros(num_columns, dtype=np.int8)
    #     hole_depths_per_col = np.zeros(num_columns, dtype=np.int8)
    #     cumulative_wells_per_col = np.zeros(num_columns, dtype=np.int8)
    #     # row_transitions = 0
    #     for col_ix, lowest_free_row in enumerate(lowest_free_rows):
    #         # There is at least one column_transition from the highest full cell (or the bottom which is assumed to be full) to "the top".
    #         col_transitions_per_col[col_ix] += 1
    #         if col_ix == 0:
    #             local_well_streak = 0
    #             if lowest_free_row > 0:
    #                 col = representation[:lowest_free_row, col_ix]
    #                 cell_below = 1
    #
    #                 # Needed for hole_depth
    #                 # TODO: Optimize... only count the first time when an actual hole is found
    #                 number_of_full_cells_above = numba_sum_int(col)
    #
    #                 for row_ix, cell in enumerate(col):
    #                     if cell == 0:
    #                         # Holes
    #                         holes_per_col[col_ix] += 1
    #                         rows_with_holes_set.add(row_ix)
    #                         hole_depths_per_col[col_ix] += number_of_full_cells_above
    #
    #                         # Column transitions
    #                         if cell_below:
    #                             col_transitions_per_col[col_ix] += 1
    #
    #                         # Row transitions and wells
    #                         # Because col_ix == 0, all left_cells are 1
    #                         # row_transitions += 1
    #                         row_transitions_per_col[col_ix] += 1
    #                         if representation[row_ix, col_ix + 1]:  # if cell to the right is full
    #                             local_well_streak += 1
    #                             cumulative_wells_per_col[col_ix] += local_well_streak
    #                         else:
    #                             local_well_streak = 0
    #
    #                     else:  # cell is 1!
    #                         local_well_streak = 0
    #
    #                         # Keep track of full cells above for hole_depth-feature
    #                         number_of_full_cells_above -= 1
    #
    #                         # Column transitions
    #                         if not cell_below:
    #                             col_transitions_per_col[col_ix] += 1
    #
    #                     # Define 'cell_below' for next (higher!) cell.
    #                     cell_below = cell
    #
    #             # Check wells until lowest_free_row_right
    #             # Check transitions until lowest_free_row_left
    #             max_well_possibility = lowest_free_rows[col_ix + 1]
    #             if max_well_possibility > lowest_free_row:
    #                 for row_ix in range(lowest_free_row, max_well_possibility):
    #                     if representation[row_ix, col_ix + 1]:  # if cell to the right is full
    #                         local_well_streak += 1
    #                         cumulative_wells_per_col[col_ix] += local_well_streak
    #                     else:
    #                         local_well_streak = 0
    #             # # Add row transitions for each empty cell above lowest_free_row
    #             row_transitions_per_col[col_ix] += (num_rows - lowest_free_row)
    #
    #         elif col_ix == num_columns - 1:
    #             local_well_streak = 0
    #             if lowest_free_row > 0:
    #                 col = representation[:lowest_free_row, col_ix]
    #                 cell_below = 1
    #
    #                 # Needed for hole_depth
    #                 number_of_full_cells_above = numba_sum_int(col)
    #
    #                 for row_ix, cell in enumerate(col):
    #                     if cell == 0:
    #                         # Holes
    #                         holes_per_col[col_ix] += 1
    #                         rows_with_holes_set.add(row_ix)
    #                         hole_depths_per_col[col_ix] += number_of_full_cells_above
    #
    #                         # Column transitions
    #                         if cell_below:
    #                             col_transitions_per_col[col_ix] += 1
    #
    #                         # Wells and row transitions
    #                         # Because this is the last column (the right border is "full") and cell == 0:
    #                         row_transitions_per_col[col_ix + 1] += 1
    #                         if representation[row_ix, col_ix - 1]:  # if cell to the left is full
    #                             row_transitions_per_col[col_ix] += 1
    #                             local_well_streak += 1
    #                             cumulative_wells_per_col[col_ix] += local_well_streak
    #                         else:
    #                             local_well_streak = 0
    #
    #                     else:  # cell is 1!
    #                         local_well_streak = 0
    #
    #                         # Keep track of full cells above for hole_depth-feature
    #                         number_of_full_cells_above -= 1
    #
    #                         # Column transitions
    #                         if not cell_below:
    #                             col_transitions_per_col[col_ix] += 1
    #
    #                         # Row transitions
    #                         cell_left = representation[row_ix, col_ix - 1]
    #                         if not cell_left:
    #                             row_transitions_per_col[col_ix] += 1
    #
    #                     # Define 'cell_below' for next (higher!) cell.
    #                     cell_below = cell
    #
    #             # Check wells until minimum(lowest_free_row_left, lowest_free_row_right)
    #             # Check transitions until lowest_free_row_left
    #             max_well_possibility = lowest_free_rows[col_ix - 1]
    #             if max_well_possibility > lowest_free_row:
    #                 for row_ix in range(lowest_free_row, max_well_possibility):
    #                     if representation[row_ix, col_ix - 1]:  # if cell to the left is full
    #                         row_transitions_per_col[col_ix] += 1
    #                         local_well_streak += 1
    #                         cumulative_wells_per_col[col_ix] += local_well_streak
    #                     else:
    #                         local_well_streak = 0
    #             # # Add row transitions from last column to border
    #             row_transitions_per_col[col_ix+1] += (num_rows - lowest_free_row)
    #         else:
    #             local_well_streak = 0
    #             if lowest_free_row > 0:
    #                 col = representation[:lowest_free_row, col_ix]
    #                 cell_below = 1
    #
    #                 # Needed for hole_depth
    #                 number_of_full_cells_above = numba_sum_int(col)
    #
    #                 for row_ix, cell in enumerate(col):
    #                     if cell == 0:
    #                         # Holes
    #                         holes_per_col[col_ix] += 1
    #                         rows_with_holes_set.add(row_ix)
    #                         hole_depths_per_col[col_ix] += number_of_full_cells_above
    #
    #                         # Column transitions
    #                         if cell_below:
    #                             col_transitions_per_col[col_ix] += 1
    #
    #                         # Wells and row transitions
    #                         cell_left = representation[row_ix, col_ix - 1]
    #                         if cell_left:
    #                             row_transitions_per_col[col_ix] += 1
    #                             cell_right = representation[row_ix, col_ix + 1]
    #                             if cell_right:
    #                                 local_well_streak += 1
    #                                 cumulative_wells_per_col[col_ix] += local_well_streak
    #                             else:
    #                                 local_well_streak = 0
    #                         else:
    #                             local_well_streak = 0
    #
    #                     else:  # cell is 1!
    #                         local_well_streak = 0
    #                         # Keep track of full cells above for hole_depth-feature
    #                         number_of_full_cells_above -= 1
    #
    #                         # Column transitions
    #                         if not cell_below:
    #                             col_transitions_per_col[col_ix] += 1
    #
    #                         # Row transitions
    #                         cell_left = representation[row_ix, col_ix - 1]
    #                         if not cell_left:
    #                             row_transitions_per_col[col_ix] += 1
    #
    #                     # Define 'cell_below' for next (higher!) cell.
    #                     cell_below = cell
    #             # Check wells until minimum(lowest_free_row_left, lowest_free_row_right)
    #             # Check transitions until lowest_free_row_left
    #             lowest_free_row_left = lowest_free_rows[col_ix - 1]
    #             lowest_free_row_right = lowest_free_rows[col_ix + 1]
    #             max_well_possibility = np.minimum(lowest_free_row_left, lowest_free_row_right)
    #
    #             # Weird case distinction because max_well_possibility always "includes" lowest_free_row_left
    #             #  but lowest_free_row_left can be higher than max_well_possibility. Don't want to double count.
    #             if max_well_possibility > lowest_free_row:
    #                 for row_ix in range(lowest_free_row, max_well_possibility):
    #                     cell_left = representation[row_ix, col_ix - 1]
    #                     if cell_left:
    #                         row_transitions_per_col[col_ix] += 1
    #                         cell_right = representation[row_ix, col_ix + 1]
    #                         if cell_right:
    #                             local_well_streak += 1
    #                             cumulative_wells_per_col[col_ix] += local_well_streak
    #                         else:
    #                             local_well_streak = 0
    #                     else:
    #                         local_well_streak = 0
    #                 if lowest_free_row_left > max_well_possibility:
    #                     for row_ix in range(max_well_possibility, lowest_free_row_left):
    #                         cell_left = representation[row_ix, col_ix - 1]
    #                         if cell_left:
    #                             row_transitions_per_col[col_ix] += 1
    #             elif lowest_free_row_left > lowest_free_row:
    #                 for row_ix in range(lowest_free_row, lowest_free_row_left):
    #                     cell_left = representation[row_ix, col_ix - 1]
    #                     if cell_left:
    #                         row_transitions_per_col[col_ix] += 1
    #
    #     rows_with_holes_set.remove(100)
    #     self.array_of_rows_with_holes = np.array(list(rows_with_holes_set), dtype=np.int8)
    #     rows_with_holes = len(rows_with_holes_set)
    #     self.col_transitions_per_col = col_transitions_per_col
    #     column_transitions = np.sum(col_transitions_per_col)
    #     self.row_transitions_per_col = row_transitions_per_col
    #     row_transitions = np.sum(row_transitions_per_col)
    #     self.holes_per_col = holes_per_col
    #     holes = np.sum(holes_per_col)
    #     self.hole_depths_per_col = hole_depths_per_col
    #     hole_depth = np.sum(hole_depths_per_col)
    #     self.cumulative_wells_per_col = cumulative_wells_per_col
    #     cumulative_wells = np.sum(cumulative_wells_per_col)
    #     eroded_pieces = numba_sum_int(self.cleared_rows_relative_to_anchor * self.pieces_per_changed_row)
    #     n_cleared_lines = numba_sum_int(self.cleared_rows_relative_to_anchor)
    #     eroded_piece_cells = eroded_pieces * n_cleared_lines
    #     landing_height = self.anchor_row + self.landing_height_bonus
    #     self.features = np.array([rows_with_holes, column_transitions, holes, landing_height,
    #                               cumulative_wells, row_transitions, eroded_piece_cells,
    #                               hole_depth])
    #     # # TODO: remove after testing
    #     # assert column_transitions == np.sum(self.col_transitions_per_col)
    #     # assert holes == np.sum(self.holes_per_col)
    #     # assert hole_depth == np.sum(self.hole_depths_per_col)
    #     # assert cumulative_wells == np.sum(self.cumulative_wells_per_col)


# @njit(fastmath=True, cache=False)
# def check_terminal(representation, num_rows):
#     is_terminal = False
#     for ix in representation[num_rows]:
#         if ix:
#             is_terminal = True
#             break
#     return is_terminal


# @njit(fastmath=True, cache=False, debug=True)
# def clear_lines_jitted(changed_lines, representation, lowest_free_rows, num_columns):
#     row_sums = np.sum(representation[changed_lines, :], axis=1)
#     is_full = (row_sums == num_columns)
#     full_lines = np.where(is_full)[0]
#     n_cleared_lines = len(full_lines)
#     if n_cleared_lines > 0:
#         lines_to_clear = changed_lines[full_lines].astype(np.int8)
#         mask_keep = np.ones(len(representation), dtype=np.bool_)
#         mask_keep[lines_to_clear] = False
#         new_cols = np.zeros((n_cleared_lines, num_columns), dtype=np.bool_)
#         representation = np.vstack((representation[mask_keep], new_cols))
#         for col_ix in range(num_columns):  # col_ix = 0
#             old_lowest_free_row = lowest_free_rows[col_ix]
#             if old_lowest_free_row > lines_to_clear[-1] + 1:
#                 lowest_free_rows[col_ix] -= n_cleared_lines
#             else:
#                 lowest = 0
#                 for row_ix in range(old_lowest_free_row - n_cleared_lines - 1, -1, -1):
#                     if representation[row_ix, col_ix] == 1:
#                         lowest = row_ix + 1
#                         break
#                 lowest_free_rows[col_ix] = lowest
#     return is_full, n_cleared_lines, representation, lowest_free_rows
#

@njit(fastmath=True, cache=False)
def numba_sum_int(int_arr):
    acc = 0
    for i in int_arr:
        acc += i
    return acc


@njit(fastmath=True, cache=False)
def numba_sum(arr):
    acc = 0.
    for i in arr:
        acc += i
    return acc


# @njit(fastmath=True, cache=False)
# def minmaxavg_jitted(x):
#     maximum = x[0]
#     minimum = x[0]
#     summed = 0
#     for i in x[1:]:
#         summed += i
#         if i > maximum:
#             maximum = i
#         elif i < minimum:
#             minimum = i
#     summed = summed / len(x)
#     return minimum, maximum, summed
#


# @njit(fastmath=True, cache=True)
# def calc_lowest_free_rows(rep):
#     num_rows, n_cols = rep.shape
#     lowest_free_rows = np.zeros(n_cols, dtype=np.int8)
#     for col_ix in range(n_cols):
#         lowest = 0
#         for row_ix in range(num_rows - 1, -1, -1):
#             if rep[row_ix, col_ix] == 1:
#                 lowest = row_ix + 1
#                 break
#         lowest_free_rows[col_ix] = lowest
#     return lowest_free_rows
#
#

