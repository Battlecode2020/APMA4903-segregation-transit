import unittest

import numpy as np

from scripts.cost_matrices import race_analysis
from scripts.segregation_metrics import (
    FOUR_CATEGORIES, fryer_majority_ssi, normalize_score,
    segregation_summary, select_four_categories,
)


class SegregationMetricTests(unittest.TestCase):
    def test_categories_preserve_hispanic_overlap_and_city_denominator(self):
        data = {'population': np.array([100, 200]),
                'race_names': np.array(['asian_alone', 'black_alone', 'white_alone']),
                'race_counts': np.array([[20, 10, 70], [100, 50, 50]]),
                'race_ethnicity_names': np.array(['hispanic_latino_any_race']),
                'race_ethnicity_counts': np.array([[60], [80]])}
        names, counts, proportions = select_four_categories(data)
        self.assertEqual(names.tolist(), list(FOUR_CATEGORIES))
        np.testing.assert_array_equal(counts[:, 2], [60, 80])
        np.testing.assert_allclose(proportions[:, 2], [.6, .4])
        self.assertTrue((proportions.sum(axis=1) > 1).all())

    def test_normalization_uses_fraction_and_preserves_negatives(self):
        self.assertAlmostEqual(normalize_score(.1, .2), -.125)
        self.assertEqual(normalize_score(1, .2), 1)
        self.assertTrue(np.isnan(normalize_score(1, 1)))
        with self.assertRaises(ValueError):
            normalize_score(.3, 20)

    def test_fryer_keeps_outgroup_leakage_and_diagonal(self):
        # Renormalizing the selected single row would incorrectly produce 1.
        P = np.array([[.3, .7], [.4, .6]])
        self.assertEqual(fryer_majority_ssi(P, [.5, .2]), (.3, 1, 1))

    def test_fryer_weights_disconnected_components_by_node_count(self):
        P = np.array([[.3, .2, 0, .5], [.2, .3, 0, .5],
                      [0, 0, .8, .2], [.3, .3, .1, .3]])
        score, count, components = fryer_majority_ssi(P, [.6, .7, .8, .1])
        self.assertAlmostEqual(score, (.5 * 2 + .8) / 3)
        self.assertEqual((count, components), (3, 2))

    def test_no_majority_matches_notebook_zero_convention(self):
        self.assertEqual(fryer_majority_ssi(np.full((2, 2), .5), [.4, .1]), (0., 0, 0))

    def test_asymmetric_weights_with_reciprocal_support_are_valid(self):
        P = np.array([[.2, .8], [.1, .9]])
        self.assertAlmostEqual(fryer_majority_ssi(P, [.6, .8])[0], 1)
        with self.assertRaises(ValueError):
            fryer_majority_ssi([[1, 0], [.5, .5]], [.7, .9])

    def test_population_share_is_weighted_and_other_metrics_have_known_values(self):
        D, p = np.array([100, 300]), np.array([[.8], [.2]])
        P, A, lam, v, residual = race_analysis(np.zeros((2, 2)), D, p)
        row = segregation_summary(P, D, p, ['example'], lam, residual).iloc[0]
        self.assertEqual(row.population, 140)
        self.assertAlmostEqual(row.population_share_K, .35)  # Not unweighted .5.
        self.assertAlmostEqual(row.segregation_index, .35)
        self.assertAlmostEqual(row.segregation_index_minus_K, 0)
        self.assertAlmostEqual(row.segregation_index_normalized, 0)
        self.assertAlmostEqual(row.fryer_majority_ssi, .25)
        self.assertAlmostEqual(row.fryer_majority_ssi_minus_K, -.1)
        self.assertAlmostEqual(row.fryer_majority_ssi_normalized, (.25 - .35) / .65)
        self.assertNotAlmostEqual(row.fryer_majority_ssi_minus_K, row.fryer_majority_ssi_normalized)
        self.assertAlmostEqual(row.network_isolation, .35)
        self.assertAlmostEqual(row.residential_isolation, (80 * .8 + 60 * .2) / 140)
        self.assertAlmostEqual(row.dissimilarity_vs_rest, abs(80 / 140 - 20 / 260))

    def test_uniform_composition_has_zero_normalized_pssi(self):
        D, p = np.array([100, 900]), np.array([[.2], [.2]])
        P, A, lam, v, residual = race_analysis([[0, -2], [-1, 0]], D, p)
        row = segregation_summary(P, D, p, ['example'], lam, residual).iloc[0]
        self.assertAlmostEqual(row.segregation_index_normalized, 0)
        self.assertAlmostEqual(row.network_isolation_normalized, 0)
        self.assertAlmostEqual(row.residential_isolation_normalized, 0)
        self.assertAlmostEqual(row.dissimilarity_vs_rest, 0)
        self.assertEqual(row.fryer_status, 'no_majority_zones_zero_by_convention')
        self.assertAlmostEqual(row.fryer_majority_ssi_normalized, -.25)


if __name__ == '__main__':
    unittest.main()
