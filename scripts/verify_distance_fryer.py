"""Independent saved-output checks for distance-neighbor Fryer scores."""
import json
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from scripts.distance_fryer import OUTPUT,CITIES,FOUR_CATEGORIES,EARTH_RADIUS_KM


def verify():
    scores=pd.read_csv(OUTPUT/'scores_all_radii.csv')
    assert len(scores)==len(CITIES)*2*3*4
    assert not scores.duplicated(['city_id','node_scale','radius_km','group']).any()
    np.testing.assert_allclose(scores.fryer_ssi_minus_K,scores.fryer_ssi-scores.population_share_K,equal_nan=True,atol=1e-13)
    np.testing.assert_allclose(scores.fryer_ssi_normalized,(scores.fryer_ssi-scores.population_share_K)/(1-scores.population_share_K),equal_nan=True,atol=1e-13)
    max_residual=0.;nodes_by_scale={'block':0,'tract':0};local_row_count=0;checked_components=0
    for city in CITIES:
        counts_by_scale=[]
        for scale in ['block','tract']:
            print('Checking',city,scale,flush=True)
            folder=OUTPUT/city/scale
            nodes=pd.read_csv(folder/'nodes.csv',dtype={'node_id':str})
            n=len(nodes);nodes_by_scale[scale]+=n
            np.testing.assert_array_equal(nodes.matrix_index,np.arange(n))
            W=sparse.load_npz(folder/'adjacency_1km.npz')
            R=sparse.load_npz(folder/'neighbor_weights_1km.npz')
            assert W.shape==R.shape==(n,n) and (W!=W.T).nnz==0 and not W.diagonal().any() and (W.data==1).all()
            degree=np.asarray(W.sum(axis=1)).ravel()
            np.testing.assert_allclose(np.asarray(R.sum(axis=1)).ravel(),degree>0,atol=1e-13)
            expected=sparse.diags(np.divide(1,degree,out=np.zeros(n,float),where=degree>0))@W
            assert np.max(np.abs((expected-R).data),initial=0)<=1e-14
            # Check actual spherical edge distances independently of the KD-tree.
            rows,cols=W.nonzero();sample=np.linspace(0,len(rows)-1,min(2000,len(rows))).astype(int)
            lat=np.radians(nodes.latitude.to_numpy());lon=np.radians(nodes.longitude.to_numpy())
            i,j=rows[sample],cols[sample]
            h=np.sin((lat[i]-lat[j])/2)**2+np.cos(lat[i])*np.cos(lat[j])*np.sin((lon[i]-lon[j])/2)**2
            distances=2*EARTH_RADIUS_KM*np.arcsin(np.sqrt(np.clip(h,0,1)))
            assert (distances<=1+1e-9).all()
            # Assignment is one dominant label, with explicit tie priority.
            counts=nodes[list(FOUR_CATEGORIES)].to_numpy()
            assignment=np.full(n,'other_unclassified',dtype=object)
            for r in [0,1,3,2]:
                assignment[(counts[:,r]==counts.max(axis=1))&(counts[:,r]>0)]=FOUR_CATEGORIES[r]
            np.testing.assert_array_equal(nodes.dominant_category,assignment)
            counts_by_scale.append(nodes[['population',*FOUR_CATEGORIES]].sum().to_numpy())
            local=pd.read_csv(folder/'local_scores_1km.csv.gz',dtype={'node_id':str})
            assert len(local)==4*n;local_row_count+=len(local)
            for group in FOUR_CATEGORIES:
                loc=local[local.group.eq(group)].sort_values('matrix_index')
                np.testing.assert_array_equal(loc.node_id,nodes.node_id)
                assigned=assignment==group;eligible=assigned & (degree>0)
                np.testing.assert_array_equal(loc.assigned_to_group,assigned)
                np.testing.assert_array_equal(loc.eligible_assigned,eligible)
                np.testing.assert_array_equal(loc.all_neighbor_count,degree)
                row=scores[scores.city_id.eq(city)&scores.node_scale.eq(scale)&scores.radius_km.eq(1)&scores.group.eq(group)].iloc[0]
                K=nodes[group].sum()/nodes.population.sum()
                np.testing.assert_allclose(row.population_share_K,K,atol=1e-14)
                assert row.eligible_assigned_node_count==eligible.sum()
                assert row.assigned_degree_zero_excluded==(assigned & (degree==0)).sum()
                assert loc.loc[~eligible,'local_fryer_ssi'].isna().all()
                selected=np.flatnonzero(eligible)
                np.testing.assert_array_equal(np.load(folder/f'B_node_indices_{group}.npy'),selected)
                B=sparse.load_npz(folder/f'B_1km_{group}.npz')
                delta=B-R[selected][:,selected]
                assert np.max(np.abs(delta.data),initial=0)<=1e-14
                if not len(selected):
                    assert np.isnan(row.fryer_ssi)
                    continue
                count,labels=connected_components(W[selected][:,selected],directed=False)
                assert count==row.component_count
                np.testing.assert_array_equal(loc.loc[eligible,'component_id'],labels)
                score=0.
                for component_id,component in loc[eligible].groupby('component_id'):
                    idx=component.matrix_index.to_numpy()
                    M=R[idx][:,idx]
                    v=component.component_eigenvector.to_numpy();value=component.component_eigenvalue.iloc[0]
                    assert (v>=0).all()
                    np.testing.assert_allclose(v.sum(),1,atol=1e-11)
                    np.testing.assert_allclose(component.component_size,len(idx))
                    np.testing.assert_allclose(component.local_fryer_ssi,len(idx)*value*v,atol=1e-10)
                    residual=float(np.max(np.abs(M@v-value*v)))
                    assert residual<=1e-10
                    max_residual=max(max_residual,residual);checked_components+=1
                    if len(idx)<=12:
                        np.testing.assert_allclose(np.linalg.eigvals(M.toarray()).real.max(),value,atol=1e-11)
                    score+=len(idx)*value/len(selected)
                np.testing.assert_allclose(score,row.fryer_ssi,atol=1e-11)
                np.testing.assert_allclose(loc.loc[eligible,'local_fryer_ssi'].mean(),row.fryer_ssi,atol=1e-11)
        np.testing.assert_array_equal(*counts_by_scale)
    report={'passed':True,'cities':len(CITIES),'score_rows_all_radii':len(scores),'primary_block_rows':len(CITIES)*4,
        'nodes_by_scale':nodes_by_scale,'primary_local_score_rows':local_row_count,'checked_components':checked_components,
        'maximum_recomputed_eigenpair_residual':max_residual,
        'checks':['Block/tract demographic totals agree','Unique ordered node IDs',
            'Neighbor symmetry, no self edges and sampled actual1km distances','Equal all-neighbor degree weights',
            'Dominant assignments and explicit tie rule','Geographic isolate exclusion and other-category inclusion in denominator',
            'B index mapping and original weights','Component eigenpairs, local scaling and city aggregation',
            'Requested population-share adjustment and normalization'],
        'routes_api_calls':0}
    (OUTPUT/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    return report


if __name__=='__main__':verify()
