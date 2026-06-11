import numpy as np


def compose_component_matrices(A, B, component_ops, aggregation):
    """
    Calculates [C] = [A] * [B]
    """
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
                component_ops[k % len(component_ops)](row_from_a[k], col_from_b[k])
                for k in range(cols_A)
            ]
            result[i, j] = aggregation(combined)

    return result


def solve_component_system(A, b, impl_func, aggregator_func):
    """
    Finds vector x in equation: A * x = b
    Using Theorem 9: g_j = Aggregator_i( Implication(a_ij, b_i) )
    """
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


def compute_reduced_matrix(A, x, b, norm_func, mode='eq'):
    rows_A, cols_A = A.shape
    A_reduced = np.zeros_like(A)

    b_component = b.flatten()
    x_component = x.flatten()

    for i in range(rows_A):
        for j in range(cols_A):
            val = norm_func(A[i, j], x_component[j])

            if mode == 'eq':
                condition = np.isclose(val, b_component[i], atol=1e-6)
            elif mode == 'ge':
                condition = val >= b_component[i] - 1e-6
            else:
                raise ValueError("Mode must be 'eq' or 'ge'")

            if condition:
                A_reduced[i, j] = A[i, j]
            else:
                A_reduced[i, j] = 0.0

    return A_reduced


def binarize_reduced_matrix(A_reduced):
    return np.where(A_reduced > 0, 1.0, 0.0)


def find_minimal_component_solutions(A, b, A_reduced, di_norm_func, norm_func, mode='eq'):

    rows_A, cols_A = A.shape
    b_flat = b.flatten()

    valid_rows = [i for i in range(rows_A) if b_flat[i] > 0]
    valid_rows.sort(key=lambda x: b_flat[x], reverse=True)

    candidate_minimal_solutions = []

    def step(current_V, current_K, current_v):
        if not current_V:
            candidate_minimal_solutions.append(current_v.copy())
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
                if norm_func(A_reduced[s, k_i], v_next[k_i]) < b_flat[s] - 1e-6:
                    V_next.append(s)

            step(V_next, K_next, v_next)

    step(valid_rows, set(), np.zeros(cols_A))

    if not candidate_minimal_solutions:
        print("DEBUG: No potential vectors found.")
        return []

    minimal_component_solutions = []
    for v in candidate_minimal_solutions:
        is_minimal = True
        for other_v in candidate_minimal_solutions:
            if np.all(other_v <= v + 1e-6) and not np.allclose(v, other_v, atol=1e-6):
                is_minimal = False
                break

        if is_minimal:
            if not any(np.allclose(v, mv, atol=1e-6) for mv in minimal_component_solutions):
                minimal_component_solutions.append(v)

    if not minimal_component_solutions:
        print("DEBUG: After filtration set the of minimal vectors is empty.")

    return minimal_component_solutions

def validate_l_star_condition(membership_matrix, nonmembership_matrix, tolerance=1e-6):
    """
     mu + nu <= 1.
    """
    sum_matrix = membership_matrix + nonmembership_matrix
    is_valid = np.all(sum_matrix <= 1.0 + tolerance)
    return is_valid, sum_matrix

def combine_components_to_ifs(mu_matrix, nu_matrix):
    """
    Combination back to tuples.
    """
    rows, cols = mu_matrix.shape
    combined = np.empty((rows, cols), dtype=object)
    for i in range(rows):
        for j in range(cols):
            combined[i, j] = (mu_matrix[i, j], nu_matrix[i, j])
    return combined

def compose_ifs_matrices(A_mu, A_nu, B_mu, B_nu, membership_ops, nonmembership_ops):
    """
    Composition and output to tuples.
    """
    membership_result = compose_component_matrices(A_mu, B_mu, membership_ops, np.max)
    nonmembership_result = compose_component_matrices(A_nu, B_nu, nonmembership_ops, np.min)
    return combine_components_to_ifs(membership_result, nonmembership_result)

def solve_ifs_system_candidate(A_mu, A_nu, b_mu, b_nu, induced_implication_mu, dual_induced_implication_nu):
    """
    Finding vector and output to tuples
    """
    greatest_membership_solution = solve_component_system(A_mu, b_mu, induced_implication_mu, np.min)
    least_nonmembership_solution = solve_component_system(A_nu, b_nu, dual_induced_implication_nu, np.max)
    return combine_components_to_ifs(greatest_membership_solution, least_nonmembership_solution)