import numpy as np
import pandas as pd
import os

from muflon.io import (
    parse_data_to_matrices,
    save_separate_results_to_csv,
    save_joined_results_to_csv
)
from muflon.operations import (
    fuzzy_composition_multi,
    solve_fuzzy_vector,
    fuzzy_composition_joined,
    solve_fuzzy_vector_joined
)
from muflon.norms import get_norm


def create_dummy_data():
    df1 = pd.DataFrame([
        ["Col1", "Col2"],
        ["(0.5, 0.2)", "(0.8, 0.1)"],
        ["(0.3, 0.4)", "(0.9, 0.0)"]
    ])
    df2 = pd.DataFrame([
        ["Vector"],
        ["(0.4, 0.3)"],
        ["(0.7, 0.2)"]
    ])
    df1.to_csv('Data1.csv', sep=';', index=False, header=False)
    df2.to_csv('Data2.csv', sep=';', index=False, header=False)


def get_data_wrapper(filename, col_start, col_end, header_rows=0):
    try:
        df = pd.read_csv(filename, sep=';', header=None, skiprows=header_rows)
        df_subset = df.iloc[:, col_start:col_end]
        return parse_data_to_matrices(df_subset)
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return None, None


def test_multiplication_separate(file1, range1, header1, file2, range2, header2):
    print("\n--- TEST: Separate Multiplication ---")
    A_mu, A_nu = get_data_wrapper(file1, range1[0], range1[1], header1)
    B_mu, B_nu = get_data_wrapper(file2, range2[0], range2[1], header2)

    if A_mu is None: return

    t_norm = get_norm('T_M')
    s_conorm = get_norm('S_M')

    res_mu = fuzzy_composition_multi(A_mu, B_mu, [t_norm], np.max)
    res_nu = fuzzy_composition_multi(A_nu, B_nu, [s_conorm], np.min)

    save_separate_results_to_csv(res_mu, res_nu, "Test_Mult_Separate.csv")


def test_multiplication_joined(file1, range1, header1, file2, range2, header2):
    print("\n--- TEST: Joined Multiplication ---")
    A_mu, A_nu = get_data_wrapper(file1, range1[0], range1[1], header1)
    B_mu, B_nu = get_data_wrapper(file2, range2[0], range2[1], header2)

    if A_mu is None: return

    t_norm = get_norm('T_M')
    s_conorm = get_norm('S_M')

    res_tuples = fuzzy_composition_joined(A_mu, A_nu, B_mu, B_nu, [t_norm], [s_conorm])

    save_joined_results_to_csv(res_tuples, "Test_Mult_Joined.csv")


def test_vector_separate(file_matrix, range_matrix, header_matrix, file_vector, range_vector, header_vector):
    print("\n--- TEST: Separate Vector Finding ---")
    A_mu, A_nu = get_data_wrapper(file_matrix, range_matrix[0], range_matrix[1], header_matrix)
    b_mu, b_nu = get_data_wrapper(file_vector, range_vector[0], range_vector[1], header_vector)

    if A_mu is None: return

    imp_func_mu = get_norm('I_TM')
    imp_func_nu = get_norm('I_TL')

    res_x_mu = solve_fuzzy_vector(A_mu, b_mu, imp_func_mu, np.min)
    res_x_nu = solve_fuzzy_vector(A_nu, b_nu, imp_func_nu, np.max)

    save_separate_results_to_csv(res_x_mu, res_x_nu, "Test_Vector_Separate.csv")


def test_vector_joined(file_matrix, range_matrix, header_matrix, file_vector, range_vector, header_vector):
    print("\n--- TEST: Joined Vector Finding ---")
    A_mu, A_nu = get_data_wrapper(file_matrix, range_matrix[0], range_matrix[1], header_matrix)
    b_mu, b_nu = get_data_wrapper(file_vector, range_vector[0], range_vector[1], header_vector)

    if A_mu is None: return

    imp_func_mu = get_norm('I_TM')
    imp_func_nu = get_norm('I_TL')

    res_x_tuples = solve_fuzzy_vector_joined(A_mu, A_nu, b_mu, b_nu, imp_func_mu, imp_func_nu)

    save_joined_results_to_csv(res_x_tuples, "Test_Vector_Joined.csv")


if __name__ == "__main__":
    create_dummy_data()
    try:
        test_multiplication_separate(
            file1='Data1.csv', range1=(0, 2), header1=1,
            file2='Data2.csv', range2=(0, 1), header2=1
        )
        test_multiplication_joined(
            file1='Data1.csv', range1=(0, 2), header1=1,
            file2='Data2.csv', range2=(0, 1), header2=1
        )
        test_vector_separate(
            file_matrix='Data1.csv', range_matrix=(0, 2), header_matrix=1,
            file_vector='Data2.csv', range_vector=(0, 1), header_vector=1
        )
        test_vector_joined(
            file_matrix='Data1.csv', range_matrix=(0, 2), header_matrix=1,
            file_vector='Data2.csv', range_vector=(0, 1), header_vector=1
        )
    except Exception as e:
        print(f"Execution failed: {e}")