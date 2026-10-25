import unittest
from types import SimpleNamespace
import numpy as np
import pandas as pd
from scripts.cost_matrices import combine_modes, fare_class, race_analysis


def inputs(walk_minutes=20,paid_minutes=10,free_minutes=np.inf):
    return {'walk_seconds':np.array([[0,walk_minutes*60],[walk_minutes*60,0]]),
            'transit_paid_seconds':np.array([[np.inf,paid_minutes*60],[paid_minutes*60,np.inf]]),
            'transit_free_seconds':np.array([[np.inf,free_minutes*60],[free_minutes*60,np.inf]]),
            'transit_proxy_seconds':np.array([[0,1800],[1800,0]])}


class CostMatrixTests(unittest.TestCase):
    def test_slower_walk_can_be_cheaper_than_faster_train(self):
        result=combine_modes(inputs(),omega=15,transit_fare=3)
        self.assertEqual(result['chosen_mode'][0,1],'WALK')
        self.assertEqual(result['C_usd'][0,1],5)
        self.assertEqual(result['transit_generalized_cost_usd'][0,1],5.5)
        self.assertAlmostEqual(result['F'][0,1],np.exp(-.5))

    def test_minimizes_fare_plus_time_across_transit_candidates(self):
        result=combine_modes(inputs(60,10,15),omega=15,transit_fare=3)
        self.assertEqual(result['chosen_mode'][0,1],'TRANSIT')
        self.assertEqual(result['selected_price_usd'][0,1],0)
        self.assertEqual(result['C_usd'][0,1],3.75)

    def test_floor_and_local_diagonal(self):
        result=combine_modes(inputs(1,10),c0=1.25)
        self.assertTrue((result['C_usd']==1.25).all())
        self.assertEqual(result['chosen_mode'][0,0],'LOCAL')
        self.assertEqual(result['selected_price_usd'][0,0],0)
        self.assertAlmostEqual(result['F'][0,0],np.exp(-.125))

    def test_missing_data_stays_unknown_and_proxy_is_labeled(self):
        data=inputs(np.inf,np.inf)
        strict=combine_modes(data,use_transit_proxy=False)
        self.assertTrue(np.isnan(strict['C_usd'][0,1]))
        self.assertTrue(np.isnan(strict['F'][0,1]))
        complete=combine_modes(data,use_transit_proxy=True)
        self.assertEqual(complete['chosen_mode'][0,1],'TRANSIT')
        self.assertEqual(complete['chosen_source'][0,1],'transit_proxy')
        self.assertEqual(complete['C_usd'][0,1],10.5)

    def test_zero_value_of_time_does_not_turn_missing_modes_into_free_routes(self):
        result=combine_modes(inputs(np.inf,np.inf),omega=0,use_transit_proxy=False)
        self.assertTrue(np.isnan(result['C_usd'][0,1]))

    def test_destination_population_and_race_match_formula(self):
        F=np.array([[1,.25],[.5,1]])
        D=np.array([100,300])
        p=np.array([[.2,.8],[.7,.3]])
        P,A,lam,v,residual=race_analysis(np.log(F),D,p)
        expected=F*D[None,:]/(F@D)[:,None]
        np.testing.assert_allclose(P,expected)
        np.testing.assert_allclose(A[0],expected*p[:,0][None,:])
        np.testing.assert_allclose(v.sum(axis=0),1)
        for r in range(2):np.testing.assert_allclose(A[r]@v[:,r],lam[r]*v[:,r])
        self.assertTrue((residual<1e-12).all())

    def test_uniform_proportion_has_expected_index_and_right_vector(self):
        P,A,lam,v,residual=race_analysis(np.array([[0,-2],[-3,0]]),[1,9],np.array([[.4],[.4]]))
        self.assertAlmostEqual(lam[0],.4)
        np.testing.assert_allclose(v[:,0],[.5,.5])

    def test_sir_fare_only_at_controlled_endpoints(self):
        platforms=pd.DataFrame({'parent_station':['S20','S25','S30','S31','R01']})
        label=SimpleNamespace(legs=(('board',0,'trip','SI',0),('ride',0,1,'trip','SI',0,100)))
        self.assertEqual(fare_class(label,1,platforms),0)
        self.assertEqual(fare_class(label,2,platforms),1)
        subway=SimpleNamespace(legs=(('board',4,'trip','N',0),('ride',4,1,'trip','N',0,100)))
        self.assertEqual(fare_class(subway,1,platforms),1)
