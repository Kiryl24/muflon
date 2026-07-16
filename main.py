import numpy as np
import pandas as pd

from muflon.data_io import (
    parse_ifs_csv_to_components,
    save_component_results_as_ifs_csv,
    save_joined_results_to_csv,
)

from muflon.ifs_operators import get_operator

from muflon.ifs_operations import (
    compose_component_matrices,
    solve_component_system,
    combine_components_to_ifs,
    validate_l_star_condition,
)


TOLERANCE = 1e-6


def create_dummy_data():
    """
    Original example from the old main.py.

    Data1.csv is a 2x2 intuitionistic fuzzy matrix A.
    Data2.csv is a 2x1 intuitionistic fuzzy vector.
    """
    df1 = pd.DataFrame([
        ["Col1", "Col2"],
        ["(0.5, 0.2)", "(0.8, 0.1)"],
        ["(0.3, 0.4)", "(0.9, 0.0)"],
    ])

    df2 = pd.DataFrame([
        ["Vector"],
        ["(0.4, 0.3)"],
        ["(0.7, 0.2)"],
    ])

    df1.to_csv("Data1.csv", sep=";", index=False, header=False)
    df2.to_csv("Data2.csv", sep=";", index=False, header=False)


def load_ifs_csv(filename, col_start=None, col_end=None, header_rows=0):
    df = pd.read_csv(filename, sep=";", header=None, skiprows=header_rows)

    if col_start is not None or col_end is not None:
        df = df.iloc[:, col_start:col_end]

    membership_matrix, nonmembership_matrix = parse_ifs_csv_to_components(df)

    is_valid, sums = validate_l_star_condition(
        membership_matrix,
        nonmembership_matrix,
    )

    if not is_valid:
        raise ValueError(
            f"Input file {filename} violates L*: mu + nu <= 1.\n"
            f"mu + nu =\n{sums}"
        )

    return membership_matrix, nonmembership_matrix


def verify_component_solution(
    A_component,
    x_component,
    b_component,
    component_operation,
    aggregation,
    component_name,
):
    """
    Verifies whether the computed candidate satisfies A o x = b.
    """
    recomputed_b = compose_component_matrices(
        A_component,
        x_component,
        component_ops=[component_operation],
        aggregation=aggregation,
    )

    is_solution = np.allclose(
        recomputed_b,
        b_component,
        atol=TOLERANCE,
    )

    if not is_solution:
        print(f"\nNo solution for the {component_name} component.")
        print("Candidate x:")
        print(x_component)
        print("A o x:")
        print(recomputed_b)
        print("Expected b:")
        print(b_component)

    return is_solution, recomputed_b


def test_composition_separate():
    print("\n--- TEST: Separate Composition ---")

    A_membership, A_nonmembership = load_ifs_csv(
        "Data1.csv",
        col_start=0,
        col_end=2,
        header_rows=1,
    )

    B_membership, B_nonmembership = load_ifs_csv(
        "Data2.csv",
        col_start=0,
        col_end=1,
        header_rows=1,
    )

    membership_operation = get_operator("T_M")
    nonmembership_operation = get_operator("S_M")

    membership_result = compose_component_matrices(
        A_membership,
        B_membership,
        component_ops=[membership_operation],
        aggregation=np.max,
    )

    nonmembership_result = compose_component_matrices(
        A_nonmembership,
        B_nonmembership,
        component_ops=[nonmembership_operation],
        aggregation=np.min,
    )

    is_valid, sums = validate_l_star_condition(
        membership_result,
        nonmembership_result,
    )

    if not is_valid:
        raise ValueError(
            "The composition result violates L*: mu + nu <= 1.\n"
            f"mu + nu =\n{sums}"
        )

    print("Membership result:")
    print(membership_result)

    print("Non-membership result:")
    print(nonmembership_result)

    save_component_results_as_ifs_csv(
        membership_result,
        nonmembership_result,
        "Test_Composition_Separate.csv",
    )

    return membership_result, nonmembership_result


def test_composition_joined():
    print("\n--- TEST: Joined Composition ---")

    membership_result, nonmembership_result = test_composition_separate()

    ifs_result = combine_components_to_ifs(
        membership_result,
        nonmembership_result,
    )

    print("Joined intuitionistic fuzzy result:")
    print(ifs_result)

    save_joined_results_to_csv(
        ifs_result,
        "Test_Composition_Joined.csv",
    )

    return ifs_result


def test_system_candidate_separate():
    print("\n--- TEST: Separate System Candidate ---")

    A_membership, A_nonmembership = load_ifs_csv(
        "Data1.csv",
        col_start=0,
        col_end=2,
        header_rows=1,
    )

    b_membership, b_nonmembership = load_ifs_csv(
        "Data2.csv",
        col_start=0,
        col_end=1,
        header_rows=1,
    )

    membership_operation = get_operator("T_M")
    nonmembership_operation = get_operator("S_M")

    membership_implication = get_operator("IMP_T_M")
    nonmembership_dual_implication = get_operator("DIMP_S_M")

    greatest_membership_solution = solve_component_system(
        A_membership,
        b_membership,
        membership_implication,
        aggregation=np.min,
    )

    least_nonmembership_solution = solve_component_system(
        A_nonmembership,
        b_nonmembership,
        nonmembership_dual_implication,
        aggregation=np.max,
    )

    print("Greatest membership solution candidate:")
    print(greatest_membership_solution)

    print("Least non-membership solution candidate:")
    print(least_nonmembership_solution)

    is_l_star_valid, sums = validate_l_star_condition(
        greatest_membership_solution,
        least_nonmembership_solution,
    )

    if not is_l_star_valid:
        raise ValueError(
            "The candidate violates L*: mu + nu <= 1.\n"
            f"mu + nu =\n{sums}"
        )

    is_membership_solution, _ = verify_component_solution(
        A_component=A_membership,
        x_component=greatest_membership_solution,
        b_component=b_membership,
        component_operation=membership_operation,
        aggregation=np.max,
        component_name="membership",
    )

    is_nonmembership_solution, _ = verify_component_solution(
        A_component=A_nonmembership,
        x_component=least_nonmembership_solution,
        b_component=b_nonmembership,
        component_operation=nonmembership_operation,
        aggregation=np.min,
        component_name="non-membership",
    )

    is_full_solution = is_membership_solution and is_nonmembership_solution

    if is_full_solution:
        save_component_results_as_ifs_csv(
            greatest_membership_solution,
            least_nonmembership_solution,
            "Test_System_Solution_Separate.csv",
        )
        print("The candidate is a valid solution.")
    else:
        print(
            "\nThe candidate is not a solution of the decomposed system. "
            "It will not be saved as a confirmed solution."
        )

    return (
        greatest_membership_solution,
        least_nonmembership_solution,
        is_full_solution,
    )


def test_system_candidate_joined():
    print("\n--- TEST: Joined System Candidate ---")

    (
        greatest_membership_solution,
        least_nonmembership_solution,
        is_full_solution,
    ) = test_system_candidate_separate()

    ifs_candidate = combine_components_to_ifs(
        greatest_membership_solution,
        least_nonmembership_solution,
    )

    print("Joined intuitionistic fuzzy candidate:")
    print(ifs_candidate)

    if is_full_solution:
        save_joined_results_to_csv(
            ifs_candidate,
            "Test_System_Solution_Joined.csv",
        )
    else:
        save_joined_results_to_csv(
            ifs_candidate,
            "Test_System_Candidate_Not_Solution.csv",
        )
        print(
            "The candidate was saved only for inspection, "
            "not as a confirmed solution."
        )

    return ifs_candidate, is_full_solution


def main():
    create_dummy_data()

    test_composition_separate()
    test_composition_joined()

    test_system_candidate_separate()
    test_system_candidate_joined()


if __name__ == "__main__":
    main()