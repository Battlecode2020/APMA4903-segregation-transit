import unittest

import numpy as np
from scipy import sparse

from scripts.distance_fryer import dominant_labels, group_ssi, neighbor_graph, component_pair


class DistanceFryerTests(unittest.TestCase):
    def test_cross_group_neighbors_stay_in_degree_denominator(self):
        W=sparse.csr_matrix([[0,1,0],[1,0,1],[0,1,0]])
        result,local,_=group_ssi(W,[.7,.6,.1],[True,True,False])
        self.assertAlmostEqual(result['fryer_ssi'],np.sqrt(.5),places=12)
        self.assertAlmostEqual(local.local_fryer_ssi.mean(),np.sqrt(.5),places=12)
        self.assertTrue(np.isnan(local.local_fryer_ssi[2]))
        np.testing.assert_allclose(local.component_eigenvector[:2],np.array([np.sqrt(2),1])/(1+np.sqrt(2)))

    def test_geographic_isolate_excluded_but_cross_group_only_retained(self):
        W=sparse.csr_matrix([[0,1,0,0,0],[1,0,0,0,0],[0,0,0,1,0],[0,0,1,0,0],[0,0,0,0,0]])
        result,local,_=group_ssi(W,[.8,.8,.8,.1,.8],[True,True,True,False,True])
        self.assertEqual(result['eligible_assigned_node_count'],3)
        self.assertEqual(result['assigned_degree_zero_excluded'],1)
        self.assertAlmostEqual(result['fryer_ssi'],2/3)
        self.assertEqual(local.local_fryer_ssi[2],0)
        self.assertTrue(np.isnan(local.local_fryer_ssi[4]))

    def test_empty_group_is_undefined_not_zero(self):
        result,local,_=group_ssi(sparse.csr_matrix([[0,1],[1,0]]),[.1,.1],[False,False])
        self.assertTrue(np.isnan(result['fryer_ssi']))
        self.assertEqual(result['status'],'undefined_no_eligible_assigned_nodes')
        self.assertTrue(local.local_fryer_ssi.isna().all())

    def test_dominant_category_can_be_below_half_with_explicit_ties(self):
        # Column order: White, Black, Hispanic, Asian. Tie order: H>A>B>W.
        counts=[[40,30,20,10],[10,10,10,10],[10,10,0,10],[10,10,0,0],[0,0,0,0]]
        np.testing.assert_array_equal(dominant_labels(counts),[0,2,3,1,-1])

    def test_no_self_edges_but_coincident_distinct_nodes_are_neighbors(self):
        W=neighbor_graph(np.array([40.,40.,40.1]),np.array([-75.,-75.,-75.]),1)
        np.testing.assert_array_equal(W.toarray(),[[0,1,0],[1,0,0],[0,0,0]])

    def test_sparse_pair_matches_independent_dense_eigenvalues(self):
        n=70
        W=sparse.diags([np.ones(n-1),np.ones(n-1)],[-1,1],shape=(n,n)).tocsr()
        degree=np.asarray(W.sum(axis=1)).ravel()+1  # one outside-group neighbor per node
        value,vector,residual=component_pair(W,degree)
        dense=W.toarray()/degree[:,None]
        self.assertAlmostEqual(value,np.linalg.eigvals(dense).real.max(),places=11)
        self.assertLess(residual,1e-10)
        np.testing.assert_allclose(dense@vector,value*vector,atol=1e-10)

    def test_self_links_are_rejected(self):
        with self.assertRaises(ValueError):
            group_ssi(sparse.eye(2),[1,1],[True,True])


if __name__=='__main__':unittest.main()
