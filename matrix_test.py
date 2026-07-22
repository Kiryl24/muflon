import numpy as np
from muflon.ifs_operations import (
    compute_reduced_matrix,
    find_minimal_component_solutions,
    solve_component_system
)
from muflon.ifs_operators import get_operator


def run_test_3x3():
    A = np.array([
        [0.5, 0.8, 0.2],
        [0.1, 0.5, 0.9],
        [0.4, 0.2, 0.6]
    ])
    b = np.array([[0.5], [0.5], [0.4]])

    t_m = get_operator('T_M')
    imp_t_m = get_operator('IMP_T_M')
    dimp_t_m = get_operator('DIMP_T_M')

    print("--- Testowanie układu A(3x3) * x = b(3x1) ---")

    x_greatest = solve_component_system(A, b, impl_func=imp_t_m, aggregator_func=np.min)
    print("Rozwiązanie największe:")
    print(x_greatest)

    A_reduced = compute_reduced_matrix(A, x_greatest, b, norm_func=t_m, mode='eq')
    minimal_sols = find_minimal_component_solutions(
        A, b, A_reduced,
        dual_implication_func=dimp_t_m,
        operation_func=t_m,
        mode='eq'
    )

    print("\nRozwiązania minimalne:")
    for idx, sol in enumerate(minimal_sols):
        print(f"v_{idx + 1} =", sol)


if __name__ == "__main__":
    run_test_3x3()