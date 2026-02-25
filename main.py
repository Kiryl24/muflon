import numpy as np
import pandas as pd

from muflon.io import parse_data_to_matrices, save_results_to_csv
from muflon.operations import fuzzy_composition_multi, solve_fuzzy_vector, get_reduced_matrix, get_binarized_matrix, find_minimal_vectors

from muflon.norms import get_norm, NORM_MAP


def get_data_wrapper(filename, col_start, col_end, header_rows=0):

    try:
        df = pd.read_csv(filename, sep=';', header=None, skiprows=header_rows)
        df_subset = df.iloc[:, col_start:col_end]
        return parse_data_to_matrices(df_subset)
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return None, None


def run_multiplication(file1, range1, header1, file2, range2, header2):
    print(f"\n=== RUNNING MODE: MULTIPLICATION ===")

    A_mu, A_nu = get_data_wrapper(file1, range1[0], range1[1], header_rows=header1)
    B_mu, B_nu = get_data_wrapper(file2, range2[0], range2[1], header_rows=header2)

    if A_mu is None: return

    t_norm = get_norm('T_M')  # Min
    s_conorm = get_norm('S_M')  # Max

    print("Computing Mu (First values)...")
    res_mu = fuzzy_composition_multi(A_mu, B_mu, [t_norm], np.max)

    print("Computing Nu (Second values)...")
    res_nu = fuzzy_composition_multi(A_nu, B_nu, [s_conorm], np.min)

    save_results_to_csv(res_mu, res_nu, "Result_Multiplication.csv")


def save_minimal_vectors(min_mu, min_nu, filename="Result_Vector_Min.csv"):

    max_len = max(len(min_mu), len(min_nu))
    if max_len == 0:
        print("Brak minimalnych wektorów do zapisania.")
        return

    vec_length = len(min_mu[0]) if min_mu else len(min_nu[0])
    combined_data = []

    for v_idx in range(max_len):
        col_data = []
        for row_idx in range(vec_length):
            mu_val = min_mu[v_idx][row_idx] if v_idx < len(min_mu) else 0.0
            nu_val = min_nu[v_idx][row_idx] if v_idx < len(min_nu) else 0.0
            col_data.append(f"{mu_val:.4f}, {nu_val:.4f}")
        combined_data.append(col_data)

    df = pd.DataFrame(combined_data).T
    headers = [f"v_{i + 1}" for i in range(max_len)]
    df.to_csv(filename, sep=';', index=False, header=headers)
    print(f"Zapisano rozwiązania minimalne do: {filename}")


def run_finding_vector(file_matrix, range_matrix, header_matrix, file_vector, range_vector, header_vector):
    print(f"\n=== RUNNING MODE: FINDING VECTOR ===")

    A_mu, A_nu = get_data_wrapper(file_matrix, range_matrix[0], range_matrix[1], header_matrix)
    b_mu, b_nu = get_data_wrapper(file_vector, range_vector[0], range_vector[1], header_vector)

    if A_mu is None: return

    norm_mu = get_norm('T_M')
    imp_func_mu = get_norm('I_TM')
    di_func_mu = get_norm('DI_TM')

    norm_nu = get_norm('S_M')
    imp_func_nu = get_norm('I_TL')
    di_func_nu = get_norm('DI_TL')

    print("1. Computing maximal vector (u) for Mu...")
    res_x_mu = solve_fuzzy_vector(A_mu, b_mu, imp_func_mu, np.min)

    print("1. Computing maximal vector (u) for Nu...")
    res_x_nu = solve_fuzzy_vector(A_nu, b_nu, imp_func_nu, np.max)

    save_results_to_csv(res_x_mu, res_x_nu, "Result_Vector_Max.csv")

    print("\n2. Computing reduced & binarized matrices for Mu & Nu...")
    A_red_mu = get_reduced_matrix(A_mu, res_x_mu, b_mu, norm_mu, mode='eq')
    A_red_nu = get_reduced_matrix(A_nu, res_x_nu, b_nu, norm_nu, mode='eq')

    A_bin_mu = get_binarized_matrix(A_red_mu)
    A_bin_nu = get_binarized_matrix(A_red_nu)

    print("\n3. Finding minimal vectors (S^0) via Algorithm I/I'...")
    min_vectors_mu = find_minimal_vectors(A_mu, b_mu, A_red_mu, di_func_mu, norm_mu, mode='eq')
    min_vectors_nu = find_minimal_vectors(A_nu, b_nu, A_red_nu, di_func_nu, norm_nu, mode='eq')

    print(f" - Found {len(min_vectors_mu)} minimal vector(s) for Mu.")
    for idx, vec in enumerate(min_vectors_mu):
        print(f"   v_mu^{idx + 1} = {vec}")

    print(f" - Found {len(min_vectors_nu)} minimal vector(s) for Nu.")
    for idx, vec in enumerate(min_vectors_nu):
        print(f"   v_nu^{idx + 1} = {vec}")

    save_minimal_vectors(min_vectors_mu, min_vectors_nu, "Result_Vector_Min.csv")


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
    except Exception as e:
        print(f"Execution failed: {e}")