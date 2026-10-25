"""Verify saved paired results against source arrays and defining equations."""
import json

import numpy as np
import pandas as pd

from scripts.multicity_analysis import BASE, CITIES, PARAMETERS
from scripts.segregation_metrics import FOUR_CATEGORIES
from scripts.transit_comparison import OUTPUT, METRICS, SCENARIOS, compare_results


def verify():
    scores=pd.read_csv(OUTPUT/'scenario_scores.csv')
    pairs=pd.read_csv(OUTPUT/'paired_comparison.csv')
    tracts=pd.read_csv(OUTPUT/'all_tract_scores.csv',dtype={'tract_geoid':str})
    assert len(scores)==len(CITIES)*len(FOUR_CATEGORIES)*len(SCENARIOS)
    assert not scores.duplicated(['city_id','group','scenario']).any()
    pd.testing.assert_frame_equal(pairs,compare_results(scores),check_exact=False,atol=1e-12,rtol=1e-12)
    np.testing.assert_allclose(scores.segregation_index_normalized,
        (scores.segregation_index-scores.population_share_K)/(1-scores.population_share_K),atol=1e-12)
    np.testing.assert_allclose(scores.fryer_majority_ssi_minus_K,
        scores.fryer_majority_ssi-scores.population_share_K,atol=1e-12)
    max_residual=0.;max_fryer_residual=0.;node_count=0;transit_pairs=0
    for city in CITIES:
        print('Verifying',city,flush=True)
        nodes=pd.read_csv(BASE/city/'demographics/nodes.csv',dtype={'tract_geoid':str})
        n=len(nodes);node_count+=n
        with np.load(BASE/city/'travel/modes.npz',allow_pickle=False) as saved:
            walking=saved['walk_seconds']
            transit=saved['transit_seconds']
        previous_cost=None
        for scenario in SCENARIOS:
            out=OUTPUT/city/scenario
            local=tracts[tracts.city_id.eq(city)&tracts.scenario.eq(scenario)]
            assert len(local)==n*4
            with np.load(out/'cost_matrices.npz',allow_pickle=False) as costs, np.load(out/'analysis_matrices.npz',allow_pickle=False) as matrices:
                np.testing.assert_array_equal(matrices['tract_geoid'],nodes.tract_geoid)
                np.testing.assert_array_equal(costs['tract_geoid'],nodes.tract_geoid)
                np.testing.assert_array_equal(matrices['population'],nodes.population)
                np.testing.assert_array_equal(matrices['counts'],nodes[list(FOUR_CATEGORIES)])
                expected_walk=walking*PARAMETERS['omega_usd_per_hour']/3600
                expected_transit=transit*PARAMETERS['omega_usd_per_hour']/3600+PARAMETERS['standardized_transit_fare_usd']
                if scenario=='walking_only':expected_transit[:]=np.inf
                expected_C=np.maximum(PARAMETERS['c0_usd'],np.minimum(expected_walk,expected_transit))
                np.fill_diagonal(expected_C,PARAMETERS['c0_usd'])
                np.testing.assert_allclose(costs['walk_cost_usd'],expected_walk,atol=1e-12)
                np.testing.assert_allclose(costs['transit_cost_usd'],expected_transit,atol=1e-12)
                np.testing.assert_allclose(costs['C_usd'],expected_C,atol=1e-12)
                np.testing.assert_allclose(costs['F'],np.exp(-PARAMETERS['beta_per_usd']*expected_C),atol=1e-14)
                if previous_cost is not None:
                    assert (previous_cost<=costs['C_usd']+1e-12).all()
                    assert not (costs['chosen_mode']=='TRANSIT').any()
                else:
                    previous_cost=costs['C_usd'].copy()
                    transit_pairs+=int((costs['chosen_mode']=='TRANSIT').sum())
                D=nodes.population.to_numpy()
                expected_P=costs['F']*D[None,:]
                expected_P/=expected_P.sum(axis=1)[:,None]
                np.testing.assert_allclose(matrices['P'],expected_P,atol=1e-13)
                np.testing.assert_allclose(matrices['P'].sum(axis=1),1,atol=1e-13)
                for r,group in enumerate(FOUR_CATEGORIES):
                    row=scores[scores.city_id.eq(city)&scores.scenario.eq(scenario)&scores.group.eq(group)].iloc[0]
                    loc=local[local.group.eq(group)].sort_values('matrix_index')
                    np.testing.assert_array_equal(loc.tract_geoid,nodes.tract_geoid)
                    p=nodes[group].to_numpy()/D
                    A=costs['F']*nodes[group].to_numpy()[None,:]/(costs['F']*D[None,:]).sum(axis=1)[:,None]
                    np.testing.assert_allclose(matrices['A'][r],A,atol=1e-13)
                    v=matrices['right_eigenvectors'][:,r];value=matrices['eigenvalues'][r]
                    assert (v>=0).all()
                    np.testing.assert_allclose(v.sum(),1,atol=1e-12)
                    np.testing.assert_allclose(loc.pssi_right_eigenvector,v,atol=1e-14)
                    np.testing.assert_allclose(row.segregation_index,value,atol=1e-14)
                    np.testing.assert_allclose(row.population_share_K,nodes[group].sum()/D.sum(),atol=1e-14)
                    residual=float(np.abs(A@v-value*v).max())
                    max_residual=max(max_residual,residual)
                    assert residual<=1e-12
                    majority=loc.fryer_is_majority_tract.to_numpy()
                    np.testing.assert_array_equal(majority,p>=.5)
                    assert loc.loc[~majority,'fryer_tract_ssi'].isna().all()
                    if not majority.any():
                        assert row.fryer_majority_ssi==0
                    else:
                        np.testing.assert_allclose(loc.loc[majority,'fryer_tract_ssi'].mean(),row.fryer_majority_ssi,atol=1e-12)
                        for _,component in loc[majority].groupby('fryer_component_id'):
                            idx=component.matrix_index.to_numpy()
                            v=component.fryer_component_eigenvector.to_numpy()
                            value=component.fryer_component_eigenvalue.iloc[0]
                            np.testing.assert_allclose(v.sum(),1,atol=1e-12)
                            np.testing.assert_allclose(component.fryer_tract_ssi,len(component)*value*v,atol=1e-12)
                            residual=float(np.abs(matrices['P'][np.ix_(idx,idx)]@v-value*v).max())
                            max_fryer_residual=max(max_fryer_residual,residual)
                            assert residual<=1e-12
    assert len(tracts)==node_count*len(FOUR_CATEGORIES)*len(SCENARIOS)
    old=pd.read_csv(BASE/'city_comparison.csv').set_index(['city_id','group'])
    reference=pd.read_csv(OUTPUT/'previous_report_reference.csv').set_index(['city_id','group'])
    for metric in METRICS:
        np.testing.assert_allclose(reference[metric+'_previous_report'],old.loc[reference.index,metric],atol=1e-14)
    report={'passed':True,'cities':len(CITIES),'scenario_category_rows':len(scores),
        'paired_category_rows':len(pairs),'tracts_per_scenario':node_count,'tract_category_rows_both_scenarios':len(tracts),
        'with_transit_selected_ordered_pairs':transit_pairs,'max_pssi_eigenpair_residual':max_residual,
        'max_fryer_eigenpair_residual':max_fryer_residual,
        'checks':['Original 2020 demographic and GEOID alignment','Exactly shared walking estimates',
                  'Cost formula and transit removal','C_with <= C_without',
                  'P and A reconstructed from defining formulas','Perron vector sums and residuals',
                  'Fryer majority components, local scores and global aggregation',
                  'Requested normalization formulas','Paired differences and preserved prior-report reference']}
    (OUTPUT/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    return report


if __name__=='__main__':verify()
