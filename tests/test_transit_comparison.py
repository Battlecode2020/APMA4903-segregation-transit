import unittest

import numpy as np
import pandas as pd

from scripts.multicity_analysis import analyze_arrays
from scripts.segregation_metrics import FOUR_CATEGORIES
from scripts.transit_comparison import compare_results, paired_costs


class TransitComparisonTests(unittest.TestCase):
    def test_transit_toggle_keeps_walking_fixed_even_across_flagged_water(self):
        modes = {
            'walk_seconds': np.array([[0, 3600, 1200], [3600, 0, 4000], [1200, 4000, 0]]),
            'transit_seconds': np.array([[np.inf, 600, 600], [np.inf, np.inf, 900], [600, 900, np.inf]]),
            'walk_barrier_check_available': True,
            'walk_land_supported': np.eye(3, dtype=bool),
        }
        paired = paired_costs(modes)
        on, off = paired['with_transit'], paired['walking_only']
        np.testing.assert_array_equal(on['walk_cost_usd'], off['walk_cost_usd'])
        self.assertTrue((on['C_usd'] <= off['C_usd']).all())
        self.assertEqual(on['chosen_mode'][0, 1], 'TRANSIT')
        self.assertEqual(on['chosen_mode'][0, 2], 'WALK')
        self.assertEqual(on['chosen_source'][0, 2], 'barrier_flagged_walk_proxy')
        self.assertEqual(off['C_usd'][0, 1], 15.)
        np.testing.assert_array_equal(np.diag(off['C_usd']), 1.25)
        self.assertFalse((off['chosen_mode'] == 'TRANSIT').any())

    def test_unavailable_transit_produces_identical_costs(self):
        modes = {'walk_seconds': [[0, 3600], [3600, 0]],
                 'transit_seconds': np.full((2, 2), np.inf)}
        paired = paired_costs(modes)
        for key in ['C_usd', 'F', 'log_F', 'chosen_mode']:
            np.testing.assert_array_equal(paired['with_transit'][key], paired['walking_only'][key])

    def test_uniform_group_proportions_remove_network_effect_on_pssi(self):
        nodes = pd.DataFrame({'matrix_index': [0, 1, 2], 'tract_geoid': ['1', '2', '3'],
                              'latitude': [40., 40.1, 40.2], 'longitude': [-75., -75.1, -75.2],
                              'population': [100, 300, 200]})
        shares = [.4, .2, .3, .1]
        for name, share in zip(FOUR_CATEGORIES, shares):
            nodes[name] = (nodes.population * share).astype(int)
        modes = {'walk_seconds': [[0, 3000, 6000], [3000, 0, 3000], [6000, 3000, 0]],
                 'transit_seconds': [[np.inf, 500, 800], [600, np.inf, 600], [800, 500, np.inf]]}
        for costs in paired_costs(modes).values():
            summary, _, _ = analyze_arrays(nodes, costs)
            np.testing.assert_allclose(summary.segregation_index, shares, atol=1e-11)
            np.testing.assert_allclose(summary.segregation_index_normalized, 0, atol=1e-11)

    def test_differences_are_without_minus_with_and_population_changes_rejected(self):
        shared = dict(city_id='example', city='Example', group='white_alone',
                      populated_tract_portions=2, city_population=100, population=40,
                      population_share_K=.4, population_percent=40., majority_zone_count=1,
                      fryer_status='defined')
        rows = pd.DataFrame([
            dict(shared, scenario='with_transit', segregation_index=.5,
                 segregation_index_normalized=1/6, fryer_majority_ssi=.6, fryer_majority_ssi_minus_K=.2),
            dict(shared, scenario='walking_only', segregation_index=.7,
                 segregation_index_normalized=.5, fryer_majority_ssi=.8, fryer_majority_ssi_minus_K=.4),
        ])
        result = compare_results(rows)
        self.assertAlmostEqual(result.segregation_index_change_without_minus_with.iloc[0], .2)
        self.assertAlmostEqual(result.segregation_index_normalized_change_without_minus_with.iloc[0], 1/3)
        rows.loc[1, 'population_share_K'] = .3
        with self.assertRaisesRegex(ValueError, 'demographic property'):
            compare_results(rows)


if __name__ == '__main__':
    unittest.main()
