# MUFLON - Matrix Utility for Fuzzy Logic Operations and Norms



This open-source tool performs fuzzy matrix composition and system resolution on data loaded from CSV files. It is designed to handle **Intuitionistic Fuzzy Values** (tuples representing Membership $\mu$ and Non-membership $\nu$) and processes them through dual logic streams (T-norms for $\mu$, S-conorms for $\nu$)

## Core Concept: Tuple Processing
The system treats every data cell as a tuple $(i_1, i_2)$, representing:
1.  **Membership ($\mu$):** The first value ($i_1$).
2.  **Non-Membership ($\nu$):** The second value ($i_2$).

The script automatically splits these into two parallel calculation streams and produces **two distinct result matrices**:
* **Result 1:** Derived from the matrix of first numbers ($i_1, j_1, \dots$).
* **Result 2:** Derived from the matrix of second numbers ($i_2, j_2, \dots$).

## Data Format Requirements

The script expects CSV files with the following strict formatting rules:

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

Usage Example

The tool is orchestrated via main.py. You define the input files and column ranges for your matrices there.
Python

## 1. Perform Matrix Composition: Calculates $C = A \circ B$
### Reads columns 0-2 for Matrix A, and 0-1 for Matrix B
```python
run_multiplication(
    file1='Data1.csv', range1=(0, 2), header1=1,
    file2='Data2.csv', range2=(0, 1), header2=1
)
```

## 2. Solve System: Solves $A \circ x = b$ for separate $\mu$ and $\nu$
### Solves for vector x given Matrix A and Vector b
```python
run_finding_vector(
    file_matrix='Data1.csv', range_matrix=(0, 2), header_matrix=1,
    file_vector='Data2.csv', range_vector=(0, 1), header_vector=1
)
```
## Project Structure

### The project is divided into four distinct modules:

`main.py`: The entry point. It orchestrates the file loading, defines the execution modes (run_multiplication, run_finding_vector), and selects the specific operators (T-norms/S-conorms) to apply.

`matrices.py`: Handles raw data parsing. It contains the logic to clean string inputs and strictly split tuple strings (e.g., `0.5`, `0.2`) into two separate numerical matrices $\mu$ and $\nu$.

`operations.py`: Contains the library of mathematical operators:

**T-norms** (e.g., `T_M`, `T_P`)

**S-conorms** (e.g., `S_M`, `S_L`)

**Implications** (e.g., `I_T_M`, `I_S_M`)

`systems.py`: Contains the core algebraic algorithms for fuzzy composition (`fuzzy_composition_multi`) and vector resolution (`solve_fuzzy_vector`).

## Core Concepts & Logic
### Dual Matrix Processing

This script splits every input matrix into two parallel streams based on the tuple data:

Mu Stream ($\mu$): Uses the first value of the tuple. Processed using T-norms (e.g., Minimum) and Max aggregation.

Nu Stream ($\nu$): Uses the second value of the tuple. Processed using S-conorms (e.g., Maximum) and Min aggregation.

### Column Scoping

Data loading is controlled by parameters in get_data_from_csv (called internally by the run functions):

`col_start`: Index of the first column to read.

`col_end`: Index of the column to stop at (exclusive).

`header_rows`: Number of top rows to skip (e.g., for labels).

### Core Functions

Module: `matrices.py`

`parse_data_to_matrices(df_subset)`:

Input: A pandas DataFrame subset of strings.

Output: A tuple (`Matrix_Mu`, `Matrix_Nu`) of floats.

Module: `systems.py`

`fuzzy_composition_multi(A, B, operator_list, aggregator_func)`:

Computes the matrix product using the generalized formula: .

`solve_fuzzy_vector(A, b, impl_func, aggregator_func)`:

Solves $A \circ x = b$ using Theorem 9 (Inverse operation using implications).

### Configuration

To change the operators used (e.g., to switch from "Min/Max" to "Product/Probabilistic Sum"), edit the `run_multiplication` or `run_finding_vector` functions in `main.py`:
```Python
res_mu = fuzzy_composition_multi(A_mu, B_mu, [ops.T_P], np.max)

res_nu = fuzzy_composition_multi(A_nu, B_nu, [ops.S_P], np.min)
```