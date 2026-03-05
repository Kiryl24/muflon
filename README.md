<img alt="logo.png" height="50" src="logo.png" width="50"/> 

# MUFLON: Matrix Utility for Fuzzy Logic Operations and Norms
Muflon is a Python library designed for processing Intuitionistic Fuzzy Values (IFVs). It handles complex matrix operations by automatically splitting data into two parallel streams:

Membership (μ): Processed via T-Norms.

Non-Membership (ν): Processed via S-Conorms.

## Installation

```pip install muflon```

## Core Concept: Tuple Processing
The system treats every data cell as a tuple $(i_1, i_2)$, representing:
1.  **Membership ($\mu$):** The first value ($i_1$).
2.  **Non-Membership ($\nu$):** The second value ($i_2$).

The script automatically splits these into two parallel calculation streams and produces **two distinct result matrices**:
* **Result 1:** Derived from the matrix of first numbers ($i_1, j_1, \dots$).
* **Result 2:** Derived from the matrix of second numbers ($i_2, j_2, \dots$).

## Data Format Requirements

Muflon is designed to work with CSV files where every cell represents a tuple ($\mu$,$\nu$).

| Feature | Separator | Example | Notes |
| :--- |:----------|:---|:---|
| **Column Separator** | `;` | `col1;col2;col3` | Standard CSV delimiter for this tool. |
| **Tuple Separator** | `,` | `0.3, 0.7` | **Crucial:** Used strictly to split $\mu$ and $\nu$ values inside a cell. |
| **Decimal Point** | `.` | `0.5` | Standard float notation. |

### CSV Structure Example (`Data.csv`)
```csv
0.3, 0.7; 0.2, 0.1; 0.5, 0.9
0.7, 0.4; 0.6, 0.2; 1.0, 0.5
```
Cell `0.3`, `0.7`: The tool parses `0.3` into the Mu Matrix and `0.7` into the Nu Matrix.

Empty Tuple Values: If a cell is just `0.5`, the second value defaults to `0.0`.

## Quick Start Guide

Here is a minimal script to load data, perform a standard Max-Min composition, and save the results.

```python
import numpy as np
from muflon.io import parse_data_to_matrices, save_results_to_csv
from muflon import fuzzy_composition, solve_vector
from muflon import get_norm

# Load Data
import pandas as pd

df = pd.read_csv('data.csv', sep=';', header=None)

# Parse into Mu and Nu Matrices
# The library automatically splits the tuples for you
matrix_mu, matrix_nu = parse_data_to_matrices(df)

# Perform Composition (C = A o B)
# Mu uses Minimum T-Norm
res_mu = fuzzy_composition(matrix_mu, matrix_mu, operator='min', aggregator=np.max)

# Nu uses Maximum S-Conorm
res_nu = fuzzy_composition(matrix_nu, matrix_nu, operator='max', aggregator=np.min)

# 4. Save Results
# Generates 'output_Mu.csv' and 'output_Nu.csv'
save_results_to_csv(res_mu, res_nu, "output.csv")
```

# Available Operators

| Type                  | Code   | Alias           | Description                  |
|:----------------------|:-------|:----------------|:-----------------------------|
| **T-Norms**           | `T_M`  | `min`           | Minimum (Zadeh)              |
|                       | `T_P`  | `product`       | Algebraic Product            |
|                       | `T_L`  | `lukasiewicz`   | Bounded Difference           |
| **S-Conorms**         | `S_M`  | `max`           | Maximum                      |
|                       | `S_P`  | `probabilistic` | Probabilistic Sum            |
|                       | `S_L`     | `bounded_sum`   | Bounded Sum                  |
| **Implications**      | `I_TM`  |                 | Godel Implication            |
|                       | `I_TP`  |                 | Goguen Implication           |
|                       | `I_TL`     |            | Lukasiewicz Implication      |
| **Dual Implications** | `DI_TM`  |                 | Dual Godel Implication       |
|                       | `DI_TP`  |                 | Dual Goguen Implication      |
|                       | `DI_TL`     |            | Dual Lukasiewicz Implication |

## 1. Perform Matrix Composition: Calculates $C = A \circ B$
### Reads columns 0-2 for Matrix A, and 0-1 for Matrix B


## 2. Solve System: Solves $A \circ x = b$ for separate $\mu$ and $\nu$
### Solves for vector x given Matrix A and Vector b
```python
# Assume we have Matrix A and Vector b loaded
A_mu, A_nu = parse_data_to_matrices(df_A)
b_mu, b_nu = parse_data_to_matrices(df_b)

# Solve for Mu using Godel Implication (Induced by Min)
x_mu = solve_fuzzy_vector(A_mu, b_mu, implication='I_TM', aggregator=np.min)

# Solve for Nu using Lukasiewicz Implication (Induced by Lukasiewicz T-Norm)
x_nu = solve_fuzzy_vector(A_nu, b_nu, implication='I_TL', aggregator=np.max)
```
## Core Concepts & Logic
### Dual Matrix Processing

This script splits every input matrix into two parallel streams based on the tuple data:

Mu Stream ($\mu$): Uses the first value of the tuple. Processed using T-norms (e.g., Minimum) and Max aggregation.

Nu Stream ($\nu$): Uses the second value of the tuple. Processed using S-conorms (e.g., Maximum) and Min aggregation.

### Reduced Matrix (A′):

The reduced matrix filters out input elements that do not actively fulfill the equation constraints. By evaluating the original matrix A against the maximal solution vector (u), any cell aij that fails to satisfy the mathematical condition (e.g., aij∗uj=bi for equations) is zeroed out.

### Binarized Matrix:

This is a helpful diagnostic boolean mask built directly from the reduced matrix. It replaces all valid, preserved elements (>0) with 1.0, while keeping the rest as 0.0. This clearly highlights which columns have the potential to satisfy specific row constraints.

### Minimal Solutions Algorithm (Algorithm I/I'):
This implements a highly efficient cascading search to find the family of all minimal solution vectors (S0). The process involves:

Row Sorting: Rows where bi=0 are automatically satisfied and skipped, while the remaining rows are sorted in descending order.

Dual Implications: For each unsatisfied row, the algorithm selects a valid column and calculates the minimal required vector value using dual implications (aij←bi).

Cascading Elimination: If a selected column simultaneously satisfies other pending rows, those rows are immediately removed from the queue, preventing combinatorial explosion.

Subset Filtering: Finally, all generated candidate vectors are compared against each other. Any vector that is strictly greater than another (a superset) is discarded, leaving only the absolute minimal solutions.

### Column Scoping

Data loading is controlled by parameters in get_data_from_csv (called internally by the run functions):

`col_start`: Index of the first column to read.

`col_end`: Index of the column to stop at (exclusive).

`header_rows`: Number of top rows to skip (e.g., for labels).


### Configuration

You can define new fuzzy logic operators (T-norms, S-conorms, or Implications) in two ways:

### Option 1: The Quick Way (Script-Level)

If you are experimenting and don't want to modify the library code, you can simply define a Python function in your script and pass it directly to the composition engine.

The function must accept two arguments (`x`, `y`).

It must work with `NumPy` arrays (use `np.maximum`, `np.where`, etc., instead of standard `max` or if).

Example code:

```python
import numpy as np
from muflon import fuzzy_composition


# 1. Define your custom operator (e.g., Einstein Product)
def t_einstein(x, y):
    """Calculates (x * y) / (2 - (x + y - x*y))"""
    return (x * y) / (2 - (x + y - x * y))


# 2. Pass the function directly to the composition tool
result = fuzzy_composition(matrix_A, matrix_B, operator=t_einstein, aggregator=np.max)
```
### Option 2: The Permanent Way (Library-Level)

If you want your new operator to be part of the library (so you can call it via string like `T_EINSTEIN`), follow these steps:

Open `muflon/norms.py` Add your function definition at the end of the appropriate section (e.g., under T-NORMS).
```python
# In muflon/norms.py

def t_hamacher(x, y):
    """Hamacher Product (simplified parameter)"""
    numerator = x * y
    denominator = x + y - (x * y)
    # Avoid division by zero if both are 0
    return np.where(denominator == 0, 0, numerator / denominator)
```
Register it in NORM_MAP Scroll down to the NORM_MAP dictionary in the same file and add a key-value pair.

```python
NORM_MAP = {
    # ... previous norms ...
    'T_M': t_M,
    'T_P': t_P,
    
    # for clarity better to add new norms at the dictionary end:
    'T_HAMACHER': t_hamacher, 
}
```
Update get_norm (Optional but recommended) If you want to allow case-insensitive lookup (e.g., 'Hamacher'), add a quick alias in the get_norm function.

```python
def get_norm(identifier):
    # ... rest of function ...
    key = identifier.upper()
    
    # alias
    if key == 'HAMACHER': key = 'T_HAMACHER'
```
Now You can use your new string identifier anywhere in your project.

```python
from muflon import get_norm

res = fuzzy_composition(A, B, operator='T_HAMACHER', aggregator=np.max)
```

## Intuitionistic Fuzzy Sets (IFS) Validation
A validation mechanism has been introduced to verify whether the calculated results form valid Intuitionistic Fuzzy Sets (IFS). 
For each Membership ($\mu$) and Non-Membership ($\nu$) pair, the following condition must be met: $\mu + \nu \le 1$.
The `validate_ifs(mu_matrix, nu_matrix)` function in the `operations.py` module handles this, returning a validity flag and a matrix of sums for debugging purposes.

## Non-Membership ($\nu$) Complements
Before executing key operations (such as finding minimal vectors or generating the reduced matrix) for the Nu stream, a matrix complement calculation step has been introduced.
Every element of the $\nu$ matrix ($a_{ij}$) is transformed using the formula $1 - a_{ij}$. The resulting matrix (referred to as `Nu_prim`) is then subjected to further algebraic operations.

## Reduced Matrices & Binarization
Mechanisms for determining reduced matrices for equations and inequalities have been added to the `operations.py` module:
* `get_reduced_matrix(A, x, b, norm_func, mode)`: Generates the reduced matrix $A'_b(x)$ based on the provided solution, norm, and mode (equation `eq` or inequality `ge`).
* `get_binarized_matrix(A_reduced)`: Transforms the calculated reduced matrix into a zero-one binary matrix (all values $> 0$ become `1.0`, and the rest become `0.0`).

## Minimal Vectors Search
A tree-search algorithm has been implemented to find all minimal solutions for the system of equations/inequalities:
* `find_minimal_vectors(A, b, A_reduced, di_norm_func, norm_func, mode='eq')` 
This function relies on the reduced matrix and dual implications. It is equipped with additional debugging mechanisms that print to the console when the set of potential vectors is empty (e.g., due to a lack of valid paths for the given boundary conditions).

## Dual Implications
The operator library in the `norms.py` module has been expanded to include dual implications (required for finding minimal vectors):
* `DI_TM`: Dual implication for the Minimum norm.
* `DI_TP`: Dual implication for the Product norm.
* `DI_TL`: Dual implication for the Lukasiewicz norm.
The `NORM_MAP` registry has been updated with these keys, as well as specific operators used for research testing (e.g., `OP_EX11`, `IMP_EX11`, `DIMP_EX11`).

## Example Validation Mode
A dedicated `test_paper_example_11()` function has been added to `main.py`.It executes step-by-step mathematical calculations for a custom operation ($x \cdot \min(x,y)$), replicating the values and steps (determining the greatest solution $u$, the reduced matrix, binarization, and minimal vectors $v$) exactly as presented in Example 11 of the scientific literature.


Example usage script for library:

```python
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
```