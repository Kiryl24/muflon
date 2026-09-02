<img alt="logo.png" height="50" src="logo.png" width="50"/> 

# MUFLON: Matrix Utility for Intuitionistic Fuzzy Relational Equations

Python library for computations with intuitionistic fuzzy values and intuitionistic fuzzy relational systems of equations.

**Current version: 1.3.9**

## Installation

The package can be installed using pip:

```python
pip install muflon
```

Integrity of package can be checked using `test.py` script, based on unittest library, located in GitHub repository.

## Aviable functions
| Funtion                                                      | Description                                                                                        |
|--------------------------------------------------------------|----------------------------------------------------------------------------------------------------|
|`parse_ifs_csv_to_components(df)`|Parses a DataFrame into separate membership and non-membership matrices|
|`compose_component_matrices(A, B, component_ops, aggregation)`|Performs matrix composition for a single component using specified operations and an aggregation function|
|`combine_components_to_ifs(membership_result, nonmembership_result)`|Merges separate membership and non-membership matrices into a single matrix of intuitionistic fuzzy pairs|
|`validate_l_star_condition(membership, nonmembership)`|Validates if every pair satisfies the $L^*$ condition and returns a validity flag along with the sums|
|`compose_ifs_matrices(...)`|Computes both components and directly returns the combined intuitionistic fuzzy matrix|
|`solve_component_system(A, b, implication_operator, aggregation)`| Computes the solution candidate for a single component system using an induced implication or dual induced implication|
|`solve_ifs_system_candidate(...)`|Combines component solutions to form the greatest candidate solution for the entire intuitionistic fuzzy system|
|`compute_reduced_matrix(A, x_component, b_component, operator, mode="eq")`| Calculates the reduced matrix component-wise for finding minimal solutions|
|`binarize_reduced_matrix(A_reduced)`| Converts the reduced matrix into a binary matrix format|
|`find_minimal_component_solutions(A, b_component, A_reduced, dual_implication, operator, mode="eq")`| Determines the minimal solutions for a single component system|


## Available operators

|Meaning|Identifier|
|-|-|
|Minimum t-norm|`T_M`|
|Product t-norm|`T_P`|
|Łukasiewicz t-norm|`T_L`|
|Drastic t-norm|`T_D`|
|Fodor t-norm|`T_FD`|
|Maximum t-conorm|`S_M`|
|Probabilistic sum|`S_P`|
|Łukasiewicz t-conorm|`S_L`|
|Drastic sum|`S_D`|
|Fodor t-conorm|`S_FD`|
|Induced implication for `T_M`|`IMP_T_M`|
|Induced implication for `T_P`|`IMP_T_P`|
|Induced implication for `T_L`|`IMP_T_L`|
|Induced implication for `T_FD`|`IMP_T_FD`|
|Dual induced implication for `S_M`|`DIMP_S_M`|
|Dual induced implication for `S_P`|`DIMP_S_P`|
|Dual induced implication for `S_L`|`DIMP_S_L`|

Additional information regarding mathematical operations and example usage can be found in [manual.md](https://github.com/Kiryl24/muflon/blob/main/manual.md) in [GitHub repository.](https://github.com/Kiryl24/muflon)

TestingSources directory has prepared short dataframes to test functionality of library.