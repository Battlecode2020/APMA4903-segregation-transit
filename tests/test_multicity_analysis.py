import unittest

import numpy as np
import pandas as pd

from scripts.multicity_analysis import analyze_arrays, fryer_tract_scores, generalized_costs, perron_pair
from scripts.segregation_metrics import FOUR_CATEGORIES, fryer_majority_ssi


class MulticityAnalysisTests(unittest.TestCase):
    def test_nearly_disconnected_fallback_has_correct_perron_pair(self):
        M=np.array([[.800001,1e-10,1e-12],[1e-10,.8,1e-12],[1e-12,1e-12,.2]])
        value,vector,residual,_=perron_pair(M,max_iterations=5)
        self.assertAlmostEqual(value,np.linalg.eigvalsh(M).max(),places=12)
        np.testing.assert_allclose(M@vector,value*vector,atol=1e-12)
        self.assertTrue((vector>=0).all())
        self.assertAlmostEqual(vector.sum(),1)
        self.assertLess(residual,1e-12)

    def test_iterative_eigenpair_matches_dense_reference(self):
        rng=np.random.default_rng(42)
        M=rng.uniform(.01,1,(17,17))
        M=M/M.sum(axis=1)[:,None]*rng.uniform(.1,.9,17)[None,:]
        value,vector,residual,iterations=perron_pair(M)
        self.assertAlmostEqual(value,float(np.linalg.eigvals(M).real.max()),places=10)
        np.testing.assert_allclose(M@vector,value*vector,atol=1e-12)
        self.assertAlmostEqual(vector.sum(),1)
        self.assertTrue((vector>=0).all())

    def test_fryer_local_mean_equals_city_and_outgroup_is_undefined(self):
        P=np.array([[.3,.2,0,.5],[.2,.3,0,.5],[0,0,.8,.2],[.3,.3,.1,.3]])
        p=np.array([.6,.7,.8,.1])
        result,details=fryer_tract_scores(P,p)
        legacy=fryer_majority_ssi(P,p)
        self.assertAlmostEqual(result[0],legacy[0],places=10)
        self.assertEqual(result[1:],(3,2))
        self.assertAlmostEqual(np.nanmean(details['fryer_tract_ssi']),result[0])
        self.assertTrue(np.isnan(details['fryer_tract_ssi'][3]))
        np.testing.assert_allclose(details['fryer_component_eigenvalue'][:3],[.5,.5,.8])
        np.testing.assert_allclose(details['fryer_component_eigenvector'][:3],[.5,.5,1])

    def test_no_majority_does_not_fabricate_local_scores(self):
        result,details=fryer_tract_scores(np.full((2,2),.5),[.1,.2])
        self.assertEqual(result,(0.,0,0))
        self.assertTrue(np.isnan(details['fryer_tract_ssi']).all())

    def test_cost_uses_dollars_plus_hours_and_no_fake_transit(self):
        walk=np.array([[0,1200],[1200,0]])
        transit=np.array([[np.inf,600],[np.inf,np.inf]])
        result=generalized_costs(walk,transit)
        self.assertEqual(result['C_usd'][0,1],5)
        self.assertEqual(result['chosen_mode'][0,1],'WALK')
        self.assertEqual(result['chosen_mode'][1,0],'WALK')
        self.assertEqual(result['C_usd'][0,0],1.25)

    def test_four_category_uniform_model_and_row_alignment(self):
        nodes=pd.DataFrame({'matrix_index':[0,1],'tract_geoid':['12345678901','12345678902'],
            'latitude':[40,40.1],'longitude':[-75,-75.1],'population':[100,300]})
        for name,share in zip(FOUR_CATEGORIES,[.4,.2,.3,.1]):nodes[name]=(nodes.population*share).astype(int)
        costs=generalized_costs([[0,600],[600,0]],np.full((2,2),np.inf))
        summary,tracts,arrays=analyze_arrays(nodes,costs)
        np.testing.assert_allclose(summary.segregation_index,[.4,.2,.3,.1],atol=1e-11)
        np.testing.assert_allclose(summary.segregation_index_normalized,0,atol=1e-11)
        np.testing.assert_allclose(arrays['right_eigenvectors'],.5,atol=1e-11)
        self.assertEqual(len(tracts),8)
        np.testing.assert_array_equal(arrays['tract_geoid'],nodes.tract_geoid)

    def test_incomplete_costs_are_not_zero_connections(self):
        nodes=pd.DataFrame({'matrix_index':[0,1],'tract_geoid':['1','2'],
            'latitude':[40,40.1],'longitude':[-75,-75.1],'population':[100,300]})
        for name in FOUR_CATEGORIES:nodes[name]=nodes.population//4
        costs=generalized_costs([[0,np.inf],[np.inf,0]],np.full((2,2),np.inf))
        with self.assertRaises(ValueError):analyze_arrays(nodes,costs)


if __name__=='__main__':unittest.main()
