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

    valid_rows = [i for i in range(rows_A) if b_flat[i] > 0]

    valid_rows.sort(key=lambda x: b_flat[x], reverse=True)

    potential_results = []

    def step(current_V, current_K, current_v):
        if not current_V:
            potential_results.append(current_v.copy())
            return

        i = current_V[0]

        valid_cols = [j for j in range(cols_A) if A_reduced[i, j] > 0]

        if not valid_cols:
            return

        for k_i in valid_cols:
            v_next = current_v.copy()

            val = di_norm_func(A[i, k_i], b_flat[i])
            v_next[k_i] = max(v_next[k_i], val)

            K_next = current_K.union({k_i})

            V_next = []
            for s in current_V:
                if s == i:
                    continue

                if norm_func(A[s, k_i], v_next[k_i]) < b_flat[s] - 1e-6:
                    V_next.append(s)

            step(V_next, K_next, v_next)

    step(valid_rows, set(), np.zeros(cols_A))

    minimal_vectors = []
    for v in potential_results:
        is_minimal = True
        for other_v in potential_results:
            if np.all(other_v <= v + 1e-6) and not np.allclose(v, other_v, atol=1e-6):
                is_minimal = False
                break

        if is_minimal:
            if not any(np.allclose(v, mv, atol=1e-6) for mv in minimal_vectors):
                minimal_vectors.append(v)

    return minimal_vectors