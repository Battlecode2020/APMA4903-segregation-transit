import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from scripts.approximate_travel import Approximation, SCENARIOS, approximate_matrices, distance_matrix_km, export


class OfflineTravelTests(unittest.TestCase):
    def test_distance_units_and_coordinate_order(self):
        # One degree of longitude at latitude 60 is roughly half its equatorial distance.
        distance = distance_matrix_km([[0,0],[0,1],[60,0],[60,1]])
        self.assertAlmostEqual(distance[0,1],111.195,places=2)
        self.assertAlmostEqual(distance[2,3],55.597,places=2)
        np.testing.assert_allclose(distance,distance.T)
        np.testing.assert_array_equal(distance.diagonal(),0)

    def test_time_cost_units_and_local_convention(self):
        a = approximate_matrices([[0,0],[0,1]],Approximation(60,1,10),beta=.2,omega=12,c0=2,trip_cost_usd=3)
        self.assertAlmostEqual(a['duration_minutes'][0,1],121.19508,places=4)
        self.assertAlmostEqual(a['cost_usd'][0,1],3+12*a['duration_minutes'][0,1]/60)
        self.assertAlmostEqual(a['f'][0,0],np.exp(-.4))
        self.assertEqual(a['duration_minutes'][0,0],0)
        self.assertEqual(a['price_usd'][0,0],0)
        self.assertEqual(a['status'][0,1],'approximate')

    def test_scenarios_are_ordered(self):
        points=[[40.8,-73.9],[40.6,-74.1],[40.7,-73.7]]
        matrices=[approximate_matrices(points,p)['duration_minutes'] for p in SCENARIOS.values()]
        self.assertTrue((matrices[0] <= matrices[1]).all())
        self.assertTrue((matrices[1] <= matrices[2]).all())

    def test_bad_inputs(self):
        for parameters in [Approximation(0),Approximation(detour_factor=.5),Approximation(access_wait_minutes=-1),Approximation(float('nan'))]:
            with self.assertRaises(ValueError):
                approximate_matrices([[40,-74],[41,-73]],parameters)
        with self.assertRaises(ValueError):
            distance_matrix_km([[100,0]])

    @patch('urllib.request.urlopen',side_effect=AssertionError('No network allowed'))
    def test_export_preserves_order_without_network(self, network):
        zones=pd.DataFrame({'matrix_index':[0,1],'zone_id':['Z','A'],'latitude':[40,41],'longitude':[-74,-73],'TOTPOP':[10,20]})
        with tempfile.TemporaryDirectory() as directory:
            report=export(zones,directory)
            data=np.load(Path(directory)/'baseline/matrices.npz',allow_pickle=False)
            self.assertEqual(data['zone_ids'].tolist(),['Z','A'])
            self.assertEqual(data['duration_seconds'].shape,(2,2))
            self.assertEqual(report['routes_api_calls'],0)
            csv=pd.read_csv(Path(directory)/'baseline/duration_minutes.csv',index_col=0)
            self.assertEqual(csv.index.tolist(),['Z','A'])
            np.testing.assert_allclose(csv,data['duration_minutes'])
        network.assert_not_called()
