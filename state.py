import numpy as np


class State:
    def __init__(self, representation, lowest_free_rows=None,
                 anchor_col=0,
                 changed_lines=np.arange(1),
                 pieces_per_changed_row=np.array([0]),
                 landing_height_bonus=0.0,
                 num_features=8,
                 feature_type='bcts'):

        # Parameters
        self.representation = representation
        self.anchor_col = anchor_col
        self.pieces_per_changed_row = pieces_per_changed_row
        self.landing_height_bonus = landing_height_bonus
        self.num_features = num_features
        self.feature_type = feature_type

        # New parameters
        if lowest_free_rows is None:
            self.lowest_free_rows = calc_lowest_free_rows(self.representation)
        else:
            self.lowest_free_rows = lowest_free_rows

        self.num_rows = self.representation.shape[0]
        self.num_columns = self.representation.shape[1]

        self.n_legal_rows = self.num_rows - 4
        self.n_cleared_lines = 0
        self.anchor_row = changed_lines[0]
        self.cleared_rows_relative_to_anchor = self.clear_lines(changed_lines=changed_lines)

        self.features = None
        self.terminal_state = check_terminal(self.representation, self.n_legal_rows)  # self.is_terminal()
        self.reward = 0 if self.terminal_state else self.n_cleared_lines
        self.value_estimate = 0.0

    def __repr__(self):
        return self.print_board_to_string()

    def get_features(self, direct_by=None, order_by=None, standardize_by=None, addRBF=False):
        if self.features is None:
            self.calc_feature_values()
        features = self.features
        # if order_by is not None:
        #     features = features[order_by]
        if direct_by is not None:
            features = features * direct_by
        # if standardize_by is not None:
        #     features = features / standardize_by
        # if addRBF:
        #     features = np.append(features, np.exp(-(np.mean(self.lowest_free_rows) - np.arange(5) * self.n_legal_rows / 4)**2 / (2*(self.n_legal_rows / 5)**2)))
        return features

    def print_board(self):
        for row_ix in range(self.n_legal_rows):
            # Start from top
            row_ix = self.n_legal_rows - row_ix - 1
            print("|", end=' ')
            for col_ix in range(self.num_columns):
                if self.representation[row_ix, col_ix]:
                    print("██", end=' ')
                else:
                    print("  ", end=' ')
            print("|")

    def print_board_to_string(self):
        string = "\n"
        for row_ix in range(self.n_legal_rows):
            # Start from top
            row_ix = self.n_legal_rows - row_ix - 1
            string += "|"
            for col_ix in range(self.num_columns):
                if self.representation[row_ix, col_ix]:
                    string += "██"
                else:
                    string += "  "
            string += "|\n"
        return string

    def clear_lines(self, changed_lines):
        is_full, self.n_cleared_lines, self.representation, self.lowest_free_rows = \
            clear_lines_jitted(changed_lines,
                               self.representation,
                               self.lowest_free_rows,
                               self.num_columns)
        return is_full

    def calc_feature_values(self):
        if self.feature_type == 'bcts':
            self.calc_bcts_features()
        else:
            raise ValueError("Only 'bcts' features implemented.")

    def calc_bcts_features(self):
        features = np.zeros(self.num_features, dtype=np.float32)
        eroded_pieces = np.sum(self.cleared_rows_relative_to_anchor * self.pieces_per_changed_row)
        n_cleared_lines = np.sum(self.cleared_rows_relative_to_anchor)
        features[6] = eroded_pieces * n_cleared_lines
        features[3] = self.anchor_row + self.landing_height_bonus + 1  # I HAVE CHANGED TO BATCH BCTS WEIGHTS - DAN
        features[[0, 1, 2, 4, 5, 7]] = get_feature_values_jitted(lowest_free_rows=self.lowest_free_rows,
                                                                 representation=self.representation,
                                                                 num_rows=self.n_legal_rows,
                                                                 num_columns=self.num_columns)
        self.features = features


# @njit
def check_terminal(representation, n_legal_rows):
    is_terminal = False
    for ix in representation[n_legal_rows]:
        if ix:
            is_terminal = True
            break
    return is_terminal


# @njit
def clear_lines_jitted(changed_lines, representation, lowest_free_rows, num_columns):
    row_sums = np.sum(representation[changed_lines, :], axis=1)
    is_full = row_sums == num_columns
    full_lines = np.where(is_full)[0]
    n_cleared_lines = len(full_lines)
    if n_cleared_lines > 0:
        lines_to_clear = changed_lines[full_lines]
        mask_keep = np.ones(len(representation), dtype=np.bool_)
        mask_keep[lines_to_clear] = False
        representation = np.vstack((representation[mask_keep],
                                    np.zeros((n_cleared_lines, num_columns), dtype=np.int_)))
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
    return is_full, n_cleared_lines, representation, lowest_free_rows


# @njit
def minmaxavg_jitted(x):
    maximum = x[0]
    minimum = x[0]
    summed = 0
    for i in x[1:]:
        summed += i
        if i > maximum:
            maximum = i
        elif i < minimum:
            minimum = i
    summed = summed / len(x)
    return minimum, maximum, summed


# @njit
def calc_lowest_free_rows(rep):
    num_rows, n_cols = rep.shape
    lowest_free_rows = np.zeros(n_cols, dtype=np.int_)
    for col_ix in range(n_cols):
        lowest = 0
        for row_ix in range(num_rows - 1, -1, -1):
            if rep[row_ix, col_ix] == 1:
                lowest = row_ix + 1
                break
        lowest_free_rows[col_ix] = lowest
    return lowest_free_rows


def get_feature_values_jitted(lowest_free_rows, representation, num_rows, num_columns):
    # Adding in walls of ones either side.
    wall = np.ones((1, len(representation)))
    representation = np.concatenate((wall, representation.T, wall), axis=0).T
    lowest_free_rows_expand = np.concatenate(([num_rows], lowest_free_rows, [num_rows]))

    rows_with_holes_set = {100}
    column_transitions = 0
    holes = 0
    # landing_height
    cumulative_wells = 0
    row_transitions = 0
    # eroded_piece_cells
    hole_depth = 0

    row_transitions += num_rows - representation[:, -2].sum()  # Counting the right hand wall.

    for col_ix, lowest_free_row in zip(np.arange(len(lowest_free_rows)) + 1, lowest_free_rows):

        column_transitions += 1  # Always one column transitions to top.
        local_well_streak = 0

        if lowest_free_row > 0:  # NON EMPTY COLUMNS

            col = representation[:lowest_free_row, col_ix]
            number_of_full_cells_above = np.sum(col)  # Needed for hole_depth

            # Counting the transitions from higher left stack to current.
            if lowest_free_rows_expand[col_ix - 1] > lowest_free_rows_expand[col_ix]:
                row_transitions += abs(lowest_free_rows_expand[col_ix - 1] - lowest_free_rows_expand[col_ix])

            cell_below = 1

            for row_ix, cell in enumerate(col):

                if cell == 0:

                    # Holes
                    holes += 1

                    rows_with_holes_set.add(row_ix)
                    if col[row_ix + 1] == 1: hole_depth += number_of_full_cells_above

                    # Column transitions
                    if cell_below:
                        column_transitions += 1

                    # Wells and row transitions
                    cell_left = representation[row_ix, col_ix - 1]
                    cell_right = representation[row_ix, col_ix + 1]
                    if cell_left:
                        row_transitions += 1
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
                        column_transitions += 1

                    # Row transitions
                    cell_left = representation[row_ix, col_ix - 1]
                    if not cell_left:
                        row_transitions += 1

                # Define 'cell_below' for next (higher!) cell.
                cell_below = cell

        else:
            row_transitions += representation[:lowest_free_rows_expand[col_ix - 1], col_ix - 1].sum()

            # Check wells until minimum(lowest_free_row_left, lowest_free_row_right)
        # Check transitions until lowest_free_row_left
        lowest_free_row_left = lowest_free_rows_expand[col_ix - 1]  # I HAVE CHANGED THESE
        lowest_free_row_right = lowest_free_rows_expand[col_ix + 1]
        max_well_possibility = np.minimum(lowest_free_row_left, lowest_free_row_right)
        if max_well_possibility > lowest_free_row:
            for row_ix in range(lowest_free_row, max_well_possibility):
                cell_left = representation[row_ix, col_ix - 1]
                cell_right = representation[row_ix, col_ix + 1]
                if cell_left:
                    if cell_right:
                        local_well_streak += 1
                        cumulative_wells += local_well_streak
                    else:
                        local_well_streak = 0
                else:
                    local_well_streak = 0

    rows_with_holes_set.remove(100)
    rows_with_holes = len(rows_with_holes_set)
    # if paper_order:
    out = [rows_with_holes, column_transitions, holes, cumulative_wells, row_transitions, hole_depth]
    # else:  # ordered by standardized bcts-weights ['eroded', 'rows_with_holes', 'landing_height', 'column_transitions', 'holes', 'cumulative_wells', 'row_transitions', 'hole_depth']
    #     out = [rows_with_holes, column_transitions, holes, cumulative_wells, row_transitions, hole_depth]
    return out



