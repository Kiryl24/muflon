import unittest
import numpy as np
import pandas as pd

from muflon.data_io import parse_ifs_csv_to_components
from muflon.ifs_operators import (
    t_M, s_M, t_D, s_D, imp_T_M, imp_T_P, dual_imp_S_M, dual_imp_S_P, get_operator
)
from muflon.ifs_operations import (
    compose_component_matrices,
    solve_component_system,
    validate_l_star_condition,
    compute_reduced_matrix,
    compose_ifs_matrices,
    solve_ifs_system_candidate
)


class TestMuflonIOAdvanced(unittest.TestCase):
    def test_parse_empty_dataframe(self):
        # Oczekiwane zachowanie dla pustej ramki danych to zwrot (None, None)
        df_empty = pd.DataFrame()
        mu, nu = parse_ifs_csv_to_components(df_empty)
        self.assertIsNone(mu)
        self.assertIsNone(nu)

    def test_parse_invalid_strings(self):
        # Sprawdzenie odporności na błędne dane - parser powinien rzucić ValueError wewnętrznie
        # i zwrócić 0.0, 0.0 dla komórek, których nie da się sparsować.
        data = np.array([["invalid,string", "(0.8, 0.2)"]])
        df = pd.DataFrame(data)
        mu, nu = parse_ifs_csv_to_components(df)

        self.assertAlmostEqual(mu[0, 0], 0.0)
        self.assertAlmostEqual(nu[0, 0], 0.0)
        self.assertAlmostEqual(mu[0, 1], 0.8)


class TestMuflonOperatorsAdvanced(unittest.TestCase):
    def test_drastic_operators(self):
        # Drastyczna T-norma (t_D) - specyficzne warunki brzegowe
        self.assertAlmostEqual(float(t_D(1.0, 0.5)), 0.5)
        self.assertAlmostEqual(float(t_D(0.5, 1.0)), 0.5)
        self.assertAlmostEqual(float(t_D(0.5, 0.5)), 0.0)

        # Drastyczna S-konorma (s_D)
        self.assertAlmostEqual(float(s_D(0.0, 0.5)), 0.5)
        self.assertAlmostEqual(float(s_D(0.5, 0.0)), 0.5)
        self.assertAlmostEqual(float(s_D(0.5, 0.5)), 1.0)

    def test_division_by_zero_in_implications(self):
        # Konwersja na typy NumPy, aby uniknąć natywnego Pythonowego ZeroDivisionError
        # podczas ewaluacji obu gałęzi w funkcji np.where()
        zero_val = np.float64(0.0)
        half_val = np.float64(0.5)
        one_val = np.float64(1.0)

        self.assertAlmostEqual(float(imp_T_P(zero_val, half_val)), 1.0)
        self.assertAlmostEqual(float(imp_T_P(zero_val, zero_val)), 1.0)

        # Dualna implikacja z dzieleniem (dual_imp_S_P)
        self.assertAlmostEqual(float(dual_imp_S_P(one_val, half_val)), 0.0)
    def test_get_operator_invalid(self):
        # Podanie nieznanego operatora powinno wyrzucić ValueError
        with self.assertRaises(ValueError):
            get_operator('NIEISTNIEJACY_OPERATOR')


class TestMuflonOperationsExceptions(unittest.TestCase):
    def setUp(self):
        self.A = np.array([[0.5, 0.5], [0.5, 0.5]])
        self.B_wrong = np.array([[0.5]])  # Złe wymiary, B ma 1 wiersz, A ma 2 kolumny
        self.b = np.array([[0.5], [0.5]])

    def test_dimension_mismatch_compose(self):
        # compose_component_matrices powinno wyrzucić ValueError przy niezgodności kolumn A z wierszami B
        with self.assertRaises(ValueError) as context:
            compose_component_matrices(self.A, self.B_wrong, [t_M], np.max)
        self.assertIn("Dimension mismatch", str(context.exception))

    def test_dimension_mismatch_solve(self):
        # solve_component_system powinno wyrzucić ValueError przy niezgodności wierszy A z wierszami b
        b_wrong = np.array([[0.5]])
        with self.assertRaises(ValueError) as context:
            solve_component_system(self.A, b_wrong, imp_T_M, np.min)
        self.assertIn("Dimension mismatch", str(context.exception))

    def test_invalid_mode_reduced_matrix(self):
        # compute_reduced_matrix powinno akceptować tylko mode 'eq' lub 'ge'
        x = np.array([[0.5], [0.5]])
        with self.assertRaises(ValueError) as context:
            compute_reduced_matrix(self.A, x, self.b, t_M, mode='invalid_mode')
        self.assertIn("Mode must be 'eq' or 'ge'", str(context.exception))


class TestMuflonHighLevelWrappers(unittest.TestCase):
    def setUp(self):
        # Dane wejściowe
        self.A_mu = np.array([[0.6, 0.4], [0.1, 0.7]], dtype=float)
        self.A_nu = np.array([[0.2, 0.5], [0.8, 0.1]], dtype=float)
        self.b_mu = np.array([[0.5], [0.7]], dtype=float)
        self.b_nu = np.array([[0.3], [0.1]], dtype=float)

    def test_solve_ifs_system_candidate(self):
        # Test wysokopoziomowej funkcji rozwiązującej cały układ intuicjonistyczny naraz
        ifs_candidate = solve_ifs_system_candidate(
            self.A_mu, self.A_nu, self.b_mu, self.b_nu,
            imp_T_M, dual_imp_S_M
        )

        # Oczekujemy krotek, sprawdzamy ich kształt i strukturę
        self.assertEqual(ifs_candidate.shape, (2, 1))
        self.assertIsInstance(ifs_candidate[0, 0], tuple)

        # Walidacja rozwiązania połączonego
        mu_candidate = np.array([[ifs_candidate[0, 0][0]], [ifs_candidate[1, 0][0]]])
        nu_candidate = np.array([[ifs_candidate[0, 0][1]], [ifs_candidate[1, 0][1]]])

        is_valid, _ = validate_l_star_condition(mu_candidate, nu_candidate)
        self.assertTrue(is_valid)

    def test_compose_ifs_matrices(self):
        # Test wysokopoziomowej kompozycji intuicjonistycznej
        x_mu = np.array([[0.5], [1.0]])
        x_nu = np.array([[0.3], [0.0]])

        ifs_result = compose_ifs_matrices(
            self.A_mu, self.A_nu, x_mu, x_nu,
            [t_M], [s_M]
        )

        self.assertEqual(ifs_result.shape, (2, 1))
        # Sprawdzenie czy wynik to krotki liczb zmiennoprzecinkowych
        self.assertIsInstance(ifs_result[0, 0][0], (float, np.floating))


if __name__ == '__main__':
    unittest.main()