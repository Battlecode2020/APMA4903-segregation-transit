"""Reproduce the public city results offline from the committed snapshot.

No credentials, downloads or Routes API calls. Published figures/tables remain
unchanged; recomputed matrices, tables and optional maps go under outputs/.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import socket

import numpy as np
import pandas as pd
from scipy import sparse

from scripts.multicity_analysis import analyze_arrays, PARAMETERS
from scripts.transit_comparison import paired_costs
from scripts.distance_fryer import group_ssi, neighbor_graph
from scripts.segregation_metrics import FOUR_CATEGORIES
from scripts.walk_access_revision import CITIES

ROOT=Path(__file__).resolve().parents[1]
PUBLISHED=ROOT/'docs/outputs/multicity/walk_access_revision'
OUTPUT=ROOT/'outputs/multicity/walk_access_revision'


def offline(*args,**kwargs):
    raise RuntimeError('Networking is disabled during snapshot reproduction')


def snapshot_check():
    manifest=json.loads((ROOT/'reproducibility/snapshot_manifest.json').read_text())
    for item in manifest['files']:
        path=ROOT/item['path']
        if path.stat().st_size!=item['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:
            raise ValueError(f"Snapshot checksum failed: {item['path']}")
    return len(manifest['files'])


def compare_table(actual,reference,key,atol=1e-9):
    a=actual.set_index(key).sort_index();b=reference.set_index(key).sort_index()
    pd.testing.assert_index_equal(a.index,b.index)
    if set(a.columns)!=set(b.columns):raise ValueError('Table columns differ')
    for name in b:
        if pd.api.types.is_numeric_dtype(b[name]):
            np.testing.assert_allclose(a[name],b[name],atol=atol,rtol=0,equal_nan=True,err_msg=name)
        else:pd.testing.assert_series_equal(a[name],b[name],check_dtype=False,check_names=False)


def reproduce(make_maps=False,rebuild_neighbors=False):
    socket.socket.connect=offline
    socket.socket.connect_ex=offline
    socket.create_connection=offline
    verified=snapshot_check()
    OUTPUT.mkdir(parents=True,exist_ok=True)
    summaries=[];local_rows=[];maximum_residual=0.
    for city,name in CITIES.items():
        print('Reproducing',name,flush=True)
        folder=OUTPUT/city;folder.mkdir(parents=True,exist_ok=True)
        nodes=pd.read_csv(PUBLISHED/city/'nodes.csv',dtype={'tract_geoid':str})
        nodes.to_csv(folder/'nodes.csv',index=False)
        modepath=ROOT/'reproducibility/travel'/city/'modes.npz'
        with np.load(modepath,allow_pickle=False) as saved:modes={key:saved[key] for key in saved.files}
        np.testing.assert_array_equal(modes['tract_geoid'],nodes.tract_geoid)
        (folder/'travel').mkdir(exist_ok=True)
        shutil.copy2(PUBLISHED/city/'travel/manifest.json',folder/'travel/manifest.json')
        for scenario,costs in paired_costs(modes).items():
            summary,local,arrays=analyze_arrays(nodes,costs)
            summary=summary.drop(columns=[c for c in summary if c.startswith(('fryer_','majority_'))])
            local=local.drop(columns=[c for c in local if c.startswith('fryer_')])
            for frame in [summary,local]:
                frame.insert(0,'city',name);frame.insert(0,'city_id',city);frame.insert(0,'scenario',scenario)
                frame.insert(3,'node_scale','70 custom sections' if city=='new_york_city' else 'populated city-limit tract portions')
            summary.insert(4,'node_count',len(nodes));summary.insert(5,'city_population',int(nodes.population.sum()))
            compare_table(summary,pd.read_csv(PUBLISHED/city/scenario/'segregation_comparison.csv'),['group'])
            compare_table(local,pd.read_csv(PUBLISHED/city/scenario/'tract_scores.csv',dtype={'tract_geoid':str}),['group','tract_geoid'])
            for r in range(4):
                v=arrays['right_eigenvectors'][:,r];value=arrays['eigenvalues'][r]
                np.testing.assert_allclose(arrays['P'].sum(axis=1),1,atol=1e-12)
                np.testing.assert_allclose(v.sum(),1,atol=1e-12)
                residual=float(np.max(np.abs(arrays['A'][r]@v-value*v)))
                if residual>1e-10:raise ValueError('Eigenpair residual too large')
                maximum_residual=max(maximum_residual,residual)
            out=folder/scenario;out.mkdir(exist_ok=True)
            summary.to_csv(out/'segregation_comparison.csv',index=False)
            local.to_csv(out/'tract_scores.csv',index=False)
            np.savez_compressed(out/'analysis_matrices.npz',**arrays)
            np.savez_compressed(out/'cost_matrices.npz',**costs,tract_geoid=nodes.tract_geoid.to_numpy(dtype=str))
            report={'scenario':scenario,'city':name,'parameters':PARAMETERS,'routes_api_calls':0,
                    'source':'Committed public snapshot. All score columns and local eigenvectors checked against the published tables.'}
            (out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
            summaries.append(summary);local_rows.append(local)
    scores=pd.concat(summaries,ignore_index=True)
    scores.to_csv(OUTPUT/'scenario_scores.csv',index=False)
    pd.concat(local_rows,ignore_index=True).to_csv(OUTPUT/'all_node_scores.csv',index=False)
    expected=pd.read_csv(PUBLISHED/'distance_fryer_recomputed.csv')
    rows=[]
    for city in CITIES:
        for scale in ['block','tract']:
            folder=ROOT/'docs/outputs/distance_fryer'/city/scale
            nodes=pd.read_csv(folder/'nodes.csv',dtype={'node_id':str})
            W=sparse.load_npz(folder/'adjacency_1km.npz')
            if rebuild_neighbors:
                rebuilt=neighbor_graph(nodes.latitude.to_numpy(),nodes.longitude.to_numpy(),1.)
                if (rebuilt!=W).nnz:raise ValueError(f'Distance graph mismatch: {city}/{scale}')
            for group in FOUR_CATEGORIES:
                stats,_,_=group_ssi(W,nodes[group].to_numpy()/nodes.population.to_numpy(),nodes.dominant_category.eq(group).to_numpy())
                old=expected[expected.city_id.eq(city)&expected.fryer_node_scale.eq(scale)&expected.group.eq(group)].iloc[0]
                K=nodes[group].sum()/nodes.population.sum();value=stats['fryer_ssi']
                np.testing.assert_allclose([value,value-K,(value-K)/(1-K)],
                    old[['fryer_ssi','fryer_ssi_minus_K','fryer_ssi_normalized']].to_numpy(dtype=float),atol=1e-10,rtol=0,equal_nan=True)
                row=old.to_dict();row.update(fryer_ssi=value,fryer_ssi_minus_K=value-K,fryer_ssi_normalized=(value-K)/(1-K),
                    eigenpair_max_abs_residual=stats['max_eigenpair_residual'])
                rows.append(row)
    fryer=pd.DataFrame(rows);fryer.to_csv(OUTPUT/'distance_fryer_recomputed.csv',index=False)
    from scripts import walk_access_revision as figures
    figures.OUTPUT=OUTPUT
    figures.make_plots(scores);figures.make_fryer_plot(fryer)
    if make_maps:
        from scripts import plot_city_eigenvectors as maps
        for city in maps.CITIES:maps.render_city(city)
        maps.write_gallery()
    verification={'passed':True,'cities':len(CITIES),'scenario_category_rows':len(scores),'fryer_rows':len(fryer),
        'node_category_rows_both_scenarios':sum(len(x) for x in local_rows),'snapshot_files_verified':verified,
        'max_eigenpair_residual':maximum_residual,'maps_regenerated':bool(make_maps),
        'distance_graphs_rebuilt':bool(rebuild_neighbors),'network_calls':0,'routes_api_calls':0,
        'checks':['Committed snapshot checksums','Every PSSI summary column and local eigenvector reproduced',
                  'Normalized rows/vectors and eigenpair residuals','Both geographic Fryer scales and population-share adjustments reproduced']}
    (OUTPUT/'public_reproduction_verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    print(json.dumps(verification,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--maps',action='store_true',help='Regenerate all 60 map images')
    parser.add_argument('--rebuild-neighbors',action='store_true',help='Reconstruct the 1 km Fryer graphs from published coordinates and check exact adjacency')
    parser.add_argument('--verify-only',action='store_true',help='Check all committed snapshot file hashes without calculating')
    args=parser.parse_args()
    if args.verify_only:print('Verified snapshot files:',snapshot_check())
    else:reproduce(args.maps,args.rebuild_neighbors)
