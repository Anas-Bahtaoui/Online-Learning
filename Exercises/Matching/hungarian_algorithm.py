import numpy as np
from scipy.optimize import linear_sum_assignment


def hungarian(cost_matrix):
    """
    Hungarian algorithm.
    """
    initial_matrix = cost_matrix.copy()

    # Helpers
    def find_cols_with_single_zero(matrix):
        for i in range(matrix.shape[1]):
            if (np.sum(matrix[:, i]) == 0) == 1:
                yield i, np.argwhere(matrix[:, i] == 0).reshape(-1)[0]

    def find_rows_with_single_zero(matrix):
        for i in range(matrix.shape[0]):
            if (np.sum(matrix[i, :]) == 0) == 1:
                yield i, np.argwhere(matrix[i, :] == 0).reshape(-1)[0]

    def first_zero() -> (int, int):
        return tuple(np.argwhere(cost_matrix == 0)[0][:2])

    # Step 1: Subtract minimum of each row from all elements in the row
    for i in range(cost_matrix.shape[0]):
        cost_matrix[i, :] -= np.min(cost_matrix[i, :])

    # Step 2: Subtract minimum of each column from all elements in the column
    for j in range(cost_matrix.shape[1]):
        cost_matrix[:, j] -= np.min(cost_matrix[:, j])

    n_lines = 0
    max_length = np.max(cost_matrix.shape)
    while True:

        # Step 3: Draw lines from rows and columns to only cover 0s, have as few lines as possible.
        n_rows = cost_matrix.shape[0]
        assigned = np.array([])
        assignments = np.zeros(cost_matrix.shape, dtype=int)
        for i in range(0, n_rows):
            for j in range(0, n_rows):
                if cost_matrix[i, j] == 0 and np.sum(assignments[:, j]) == 0 and np.sum(assignments[i, :]) == 0:
                    assignments[i, j] = 1
                    assigned = np.append(assigned, i)
                    break
        rows = np.linspace(0, n_rows - 1, n_rows).astype(int)
        marked_rows = np.setdiff1d(rows, assigned)
        new_marked_rows = marked_rows.copy()
        marked_cols = np.array([])
        while len(new_marked_rows) > 0:
            new_marked_cols = np.array([], dtype=int)
            for nr in new_marked_rows:
                zeros_cols = np.argwhere(cost_matrix[nr, :] == 0).reshape(-1)
                new_marked_cols = np.append(new_marked_cols, np.setdiff1d(zeros_cols, marked_cols))
            marked_cols = np.append(marked_cols, new_marked_cols)
            new_marked_rows = np.array([], dtype=int)
            for nc in new_marked_cols:
                new_marked_rows = np.append(new_marked_rows, np.argwhere(cost_matrix[:, nc] == 1).reshape(-1))
            marked_rows = np.unique(np.append(marked_rows, new_marked_rows))
        covered_rows, covered_cols = np.setdiff1d(rows, marked_rows).astype(int), np.unique(marked_cols)
        # Step 4: Check if the number of lines drawn is same as the number of rows or columns
        n_lines = len(covered_rows) + len(covered_cols)
        if n_lines >= max_length:
            break

        # Step 5: Find the smallest entry outside of all lines and subtract it from all not crossed out and add it to all the ones crossed out
        uncovered_rows = np.setdiff1d(np.linspace(cost_matrix.shape[0] - 1, cost_matrix.shape[0]), covered_rows).astype(
            int)
        uncovered_cols = np.setdiff1d(np.linspace(cost_matrix.shape[1] - 1, cost_matrix.shape[1]), covered_cols).astype(
            int)
        min_val = np.max(cost_matrix)
        for i in uncovered_rows:
            for j in uncovered_cols:
                if cost_matrix[i, j] < min_val:
                    min_val = cost_matrix[i, j]
        for i in uncovered_rows:
            cost_matrix[i, :] -= min_val
        for j in covered_cols:
            cost_matrix[:, j] += min_val
    # Final assignment

    assignment = np.zeros(initial_matrix.shape, dtype=int)
    while True:
        # Assign single zero lines
        for i, j in find_rows_with_single_zero(cost_matrix):
            cost_matrix[i, j] += 1
            cost_matrix[:, j] += 1
            assignment[i, j] = 1
        for i, j in find_cols_with_single_zero(cost_matrix):
            cost_matrix[i, j] += 1
            cost_matrix[i, :] += 1
            assignment[i, j] = 1
        if np.sum(cost_matrix == 0) == 0:
            break
        i, j = first_zero()
        assignment[i, j] = 1
        cost_matrix[i, :] += 1
        cost_matrix[:, j] += 1
    return assignment * initial_matrix, assignment

if __name__ == '__main__':
    a = np.random.randint(100, size=(3,3))
    res = hungarian(a)
    print(f"Optimal Matching\n {res[1]} \n Value: {np.sum(res[0])}")
