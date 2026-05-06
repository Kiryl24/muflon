import pandas as pd
import numpy as np


def parse_data_to_matrices(df_subset):

    if df_subset.empty:
        return None, None

    raw_strings = df_subset.to_numpy(dtype=str)

    def parse_cell(cell):
        s = str(cell).strip()
        s = s.replace('(', '').replace(')', '').replace('[', '').replace(']', '')
        s = s.replace(' ', '')

        parts = s.split(',')

        try:
            v1 = float(parts[0])
            v2 = float(parts[1]) if len(parts) > 1 else 0.0
            return (v1, v2)
        except ValueError:

            return (0.0, 0.0)

    v_parse = np.vectorize(parse_cell, otypes=[object])
    matrix_tuples = v_parse(raw_strings)

    matrix_mu = np.vectorize(lambda x: x[0], otypes=[float])(matrix_tuples)
    matrix_nu = np.vectorize(lambda x: x[1], otypes=[float])(matrix_tuples)

    return matrix_mu, matrix_nu


def save_separate_results_to_csv(mu_matrix, nu_matrix, filename="Result_Separate.csv"):
    rows, cols = mu_matrix.shape
    combined_data = np.empty((rows, cols), dtype=object)

    for i in range(rows):
        for j in range(cols):
            combined_data[i, j] = f"{mu_matrix[i, j]:.4f}, {nu_matrix[i, j]:.4f}"

    df = pd.DataFrame(combined_data)
    df.to_csv(filename, sep=';', index=False, header=False)
    print(f"File saved (from separate matrices): {filename}")


def save_joined_results_to_csv(tuple_matrix, filename="Result_Joined.csv"):
    rows, cols = tuple_matrix.shape
    combined_data = np.empty((rows, cols), dtype=object)

    for i in range(rows):
        for j in range(cols):
            mu, nu = tuple_matrix[i, j]
            combined_data[i, j] = f"{mu:.4f}, {nu:.4f}"

    df = pd.DataFrame(combined_data)
    df.to_csv(filename, sep=';', index=False, header=False)
    print(f"File saved (from joined tuples): {filename}")