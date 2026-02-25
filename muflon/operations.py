import itertools
import numpy as np


def fuzzy_composition_multi(A, B, operator_list, aggregator_func):
    rows_A, cols_A = A.shape
    rows_B, cols_B = B.shape

    if cols_A != rows_B:
        raise ValueError(f"Dimension mismatch: A columns ({cols_A}) != B rows ({rows_B})")

    result = np.zeros((rows_A, cols_B))

    for i in range(rows_A):
        for j in range(cols_B):
            row_from_a = A[i, :]
            col_from_b = B[:, j]
            combined = [
                operator_list[k % len(operator_list)](row_from_a[k], col_from_b[k])
                for k in range(cols_A)
            ]
            result[i, j] = aggregator_func(combined)

    return result


def solve_fuzzy_vector(A, b, impl_func, aggregator_func):
    rows_A, cols_A = A.shape
    rows_b = b.shape[0]

    if rows_A != rows_b:
        raise ValueError(f"Dimension mismatch: A rows ({rows_A}) != b rows ({rows_b})")

    x_result = np.zeros(cols_A)

    for j in range(cols_A):
        column_a = A[:, j]
        implications = impl_func(column_a, b.flatten())
        x_result[j] = aggregator_func(implications)

    return x_result.reshape(-1, 1)


def get_reduced_matrix(A, x, b, norm_func, mode='eq'):
    rows_A, cols_A = A.shape
    A_prime = np.zeros_like(A)

    b_flat = b.flatten()
    x_flat = x.flatten()

    for i in range(rows_A):
        for j in range(cols_A):
            val = norm_func(A[i, j], x_flat[j])

            if mode == 'eq':
                condition = np.isclose(val, b_flat[i], atol=1e-6)
            elif mode == 'ge':
                condition = val >= b_flat[i] - 1e-6
            else:
                raise ValueError("Mode must be 'eq' or 'ge'")

            if condition:
                A_prime[i, j] = A[i, j]
            else:
                A_prime[i, j] = 0.0

    return A_prime


def get_binarized_matrix(A_reduced):
    return np.where(A_reduced > 0, 1.0, 0.0)


def find_minimal_vectors(A, b, A_reduced, di_norm_func, norm_func, mode='eq'):

    rows_A, cols_A = A.shape
    b_flat = b.flatten()

    valid_cols_per_row = []
    for i in range(rows_A):
        if b_flat[i] == 0:
            valid_cols_per_row.append([None])
        else:
            valid_cols = np.where(A_reduced[i] > 0)[0].tolist()
            if not valid_cols:
                return []
            valid_cols_per_row.append(valid_cols)

    all_combinations = list(itertools.product(*valid_cols_per_row))

    potential_vectors = []

    for combo in all_combinations:
        v = np.zeros(cols_A)
        for i, j in enumerate(combo):
            if j is None:
                continue

            val = di_norm_func(A[i, j], b_flat[i])

            v[j] = max(v[j], val)

        potential_vectors.append(v)

    valid_vectors = []
    for v in potential_vectors:
        A_v = np.zeros(rows_A)
        for i in range(rows_A):
            A_v[i] = np.max([norm_func(A[i, k], v[k]) for k in range(cols_A)])

        is_valid = True
        for i in range(rows_A):
            if mode == 'eq':
                if not np.isclose(A_v[i], b_flat[i], atol=1e-6):
                    is_valid = False;
                    break
            elif mode == 'ge':
                if A_v[i] < b_flat[i] - 1e-6:
                    is_valid = False;
                    break

        if is_valid:
            valid_vectors.append(v)

    minimal_vectors = []
    for v in valid_vectors:
        is_minimal = True
        for other_v in valid_vectors:

            if np.all(other_v <= v + 1e-6) and not np.allclose(v, other_v, atol=1e-6):
                is_minimal = False
                break

        if is_minimal:

            if not any(np.allclose(v, mv, atol=1e-6) for mv in minimal_vectors):
                minimal_vectors.append(v)

    return minimal_vectors