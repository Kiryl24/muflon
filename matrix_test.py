import numpy as np
from muflon.ifs_operations import (
    solve_component_system,
    compute_reduced_matrix,
    find_minimal_component_solutions
)
from muflon.ifs_operators import get_operator

A_mu = np.array([
    [0.4, 0.2, 0.3],
    [0.2, 0.6, 0.1],
    [0.1, 0.2, 0.5]
], dtype=float)
b_mu = np.array([[0.3], [0.6], [0.4]], dtype=float)

A_nu = np.array([
    [0.5, 0.8, 0.2],
    [0.1, 0.5, 0.9],
    [0.4, 0.2, 0.6]
], dtype=float)
b_nu = np.array([[0.3], [0.4], [0.2]], dtype=float)

t_norm = get_operator('T_M')
impl_t_norm = get_operator('IMP_T_M')

x_mu_candidate = solve_component_system(
    A_mu, b_mu, impl_func=impl_t_norm, aggregator_func=np.min
)

A_mu_reduced = compute_reduced_matrix(
    A_mu, x_mu_candidate, b_mu, norm_func=t_norm, mode='eq'
)

minimal_mu_solutions = find_minimal_component_solutions(
    A_mu, b_mu, A_mu_reduced, dual_implication_func=impl_t_norm, operation_func=t_norm, mode='eq'
)


A_nu_complement = 1.0 - A_nu
b_nu_complement = 1.0 - b_nu

x_comp_candidate = solve_component_system(
    A_nu_complement, b_nu_complement, impl_func=impl_t_norm, aggregator_func=np.min
)

A_reduced_comp = compute_reduced_matrix(
    A_nu_complement, x_comp_candidate, b_nu_complement, norm_func=t_norm, mode='eq'
)

minimal_comp_solutions = find_minimal_component_solutions(
    A_nu_complement, b_nu_complement, A_reduced_comp,
    dual_implication_func=impl_t_norm, operation_func=t_norm, mode='eq'
)

maximal_nu_solutions = [1.0 - sol for sol in minimal_comp_solutions]

print("Minimalne wektory części membership (mu):")
for idx, sol in enumerate(minimal_mu_solutions):
    print(f" u_{idx+1} = {sol.flatten()}")

print("\nOdpowiadające wektory non-membership (nu) uzyskane metodą dualności:")
for idx, sol in enumerate(maximal_nu_solutions):
    print(f" z_{idx+1} = {sol.flatten()}")