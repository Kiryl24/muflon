import numpy as np

from muflon.io import parse_data_to_matrices, save_results_to_csv
from muflon.operations import fuzzy_composition_multi, solve_fuzzy_vector, get_reduced_matrix, validate_ifs, \
    find_minimal_vectors, get_binarized_matrix
from muflon.norms import get_norm, NORM_MAP


def get_data_wrapper(filename, col_start, col_end, header_rows=0):
    """Wrapper to handle loading using your library's io module"""
    import pandas as pd
    try:
        df = pd.read_csv(filename, sep=';', header=None, skiprows=header_rows)
        df_subset = df.iloc[:, col_start:col_end]
        return parse_data_to_matrices(df_subset)
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return None, None


def run_multiplication(file1, range1, header1, file2, range2, header2):
    print(f"\nRUNNING MODE: MULTIPLICATION")

    A_mu, A_nu = get_data_wrapper(file1, range1[0], range1[1], header_rows=header1)
    B_mu, B_nu = get_data_wrapper(file2, range2[0], range2[1], header_rows=header2)

    if A_mu is None: return

    t_norm = get_norm('T_M')  # Min
    s_conorm = get_norm('S_M')  # Max

    print("Computing Mu (First values)")
    res_mu = fuzzy_composition_multi(A_mu, B_mu, [t_norm], np.max)

    print("Computing Nu (Second values)")
    res_nu = fuzzy_composition_multi(A_nu, B_nu, [s_conorm], np.min)

    save_results_to_csv(res_mu, res_nu, "Result_Multiplication.csv")


def run_finding_vector(file_matrix, range_matrix, header_matrix, file_vector, range_vector, header_vector):
    print(f"\nRUNNING MODE: FINDING VECTOR")

    A_mu, A_nu = get_data_wrapper(file_matrix, range_matrix[0], range_matrix[1], header_matrix)
    b_mu, b_nu = get_data_wrapper(file_vector, range_vector[0], range_vector[1], header_vector)

    if A_mu is None: return

    imp_func_mu = get_norm('I_TM')
    imp_func_nu = get_norm('I_TL')

    print("Computing vector x for Mu")
    res_x_mu = solve_fuzzy_vector(A_mu, b_mu, imp_func_mu, np.min)

    print("Computing vector x for Nu")
    res_x_nu = solve_fuzzy_vector(A_nu, b_nu, imp_func_nu, np.max)

    save_results_to_csv(res_x_mu, res_x_nu, "Result_Vector.csv")


def test_paper_example_11():
    print("\n=== RUNNING PAPER EXAMPLE 11 (VALIDATION) ===")

    A = np.array([
        [1.0, 0.8, 0.5, 0.75],
        [0.75, 0.8, 0.1, 1.0],
        [0.2, 0.3, 0.6, 0.4],
        [0.4, 0.5, 0.6, 0.5]
    ])
    b = np.array([[0.8], [0.6], [0.3], [0.3]])

    norm_func = get_norm('OP_EX11')
    imp_func = get_norm('IMP_EX11')
    dimp_func = get_norm('DIMP_EX11')

    u = solve_fuzzy_vector(A, b, imp_func, np.min)
    print("1. OCalculated maximal solution u:")
    print(u.flatten())

    A_prime = get_reduced_matrix(A, u, b, norm_func, mode='eq')
    print("\n2. Reduced matrix A'_b(u):")
    print(A_prime)


    A_prime_bin = get_binarized_matrix(A_prime)
    print("\n3. Binarized reduced matrix A'_b(u):")
    print(A_prime_bin)


    min_vecs = find_minimal_vectors(A, b, A_prime, dimp_func, norm_func)
    print("\n4. Calculated minimal vectors Alg(u):")
    for idx, vec in enumerate(min_vecs):
        print(f"v^{idx + 1} =", vec)


def run_ifs_validation_demo():
    print("\nRUNNING IFS VALIDATION DEMO")
    x_mu = np.array([[0.8], [0.5], [0.2]])
    x_nu = np.array([[0.1], [0.5], [0.9]])

    is_valid, sums = validate_ifs(x_mu, x_nu)

    print(f"Proper IFS? {is_valid}")
    if not is_valid:
        print("Err: sum is over 1 in example:")
        print(sums)


if __name__ == "__main__":
    try:
        run_multiplication(
            file1='Data1.csv', range1=(0, 2), header1=1,
            file2='Data2.csv', range2=(0, 1), header2=1
        )
        run_finding_vector(
            file_matrix='Data1.csv', range_matrix=(0, 2), header_matrix=1,
            file_vector='Data2.csv', range_vector=(0, 1), header_vector=1
        )

        test_paper_example_11()
        run_ifs_validation_demo()

    except Exception as e:
        print(f"Execution failed: {e}")