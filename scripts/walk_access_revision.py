"""Recompute city scores with uncapped, barrier-screened walks to/from transit.

Uses only cached Census and GTFS inputs. Preserves all previous result folders.
The geographic-neighbor Fryer model is recomputed separately and does not depend
on transit. NYC retains its 70 sections; other cities retain populated tracts.
"""
import argparse
from datetime import datetime,timezone
from html import escape
import hashlib
import json
from pathlib import Path
import time

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy import sparse

from scripts.multicity_analysis import ROOT,BASE,CITIES as ORIGINAL_CITIES,PARAMETERS,analyze_arrays,multicity_column_dictionary,tract_column_dictionary
from scripts.multicity_transit import TransitParameters,compute_city
from scripts.transit_comparison import paired_costs
from scripts.segregation_metrics import FOUR_CATEGORIES,normalize_score
from scripts.distance_fryer import group_ssi

OUTPUT=BASE/'walk_access_revision'
CITIES={**ORIGINAL_CITIES,'new_york_city':'New York City'}
PSSI_METRICS=['segregation_index','segregation_index_minus_K','segregation_index_normalized']


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def write_json_if_changed(path,value):
    text=json.dumps(value,indent=2)+'\n'
    if path.exists() and path.read_text()==text:return
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(text);temporary.replace(path)


def prepare_nodes(city):
    if city=='new_york_city':
        z=pd.read_csv(ROOT/'outputs/nyc_routes/zones.csv')
        nodes=pd.DataFrame({'matrix_index':z.matrix_index,'tract_geoid':z.zone_id,'zone_id':z.zone_id,
            'borough':z.borough,'latitude':z.latitude,'longitude':z.longitude,'population':z.TOTPOP,
            'white_alone':z.NH_WHITE+z.H_WHITE,'black_alone':z.NH_BLACK+z.H_BLACK,
            'hispanic_latino_any_race':z.HISP,'asian_alone':z.NH_ASIAN+z.H_ASIAN})
    else:nodes=pd.read_csv(BASE/city/'demographics/nodes.csv',dtype={'tract_geoid':str})
    if nodes.matrix_index.tolist()!=list(range(len(nodes))) or not nodes.tract_geoid.is_unique:
        raise ValueError('Invalid node order')
    folder=OUTPUT/city;folder.mkdir(parents=True,exist_ok=True)
    nodes.to_csv(folder/'nodes.csv',index=False)
    return nodes


def travel_inputs(city):
    if city=='new_york_city':
        # This is only a conservative chord mask, not a pedestrian road network.
        land=gpd.read_file(ROOT/'data/raw/tl_2020_36_tabblock20.shp',
            where="COUNTYFP20 IN ('005','047','061','081','085')",columns=['COUNTYFP20','ALAND20','geometry']).to_crs(32618)
        if land.empty:raise ValueError('NYC Census land filter returned no blocks')
        union=land[land.ALAND20.astype(float)>0].geometry.union_all().buffer(60)
        if union.is_empty or not union.is_valid:raise ValueError('NYC pedestrian land mask is empty or invalid')
        union=gpd.GeoSeries([union],crs=32618).to_crs(4326).iloc[0]
        return [ROOT/'data/transit/gtfs_subway.zip'],'2026-10-20',{'land_geometry':union}
    previous=json.loads((BASE/city/'travel/manifest.json').read_text())
    feeds=[]
    for feed in previous['feeds']:
        path=Path(feed['path']);path=path if path.is_absolute() else ROOT/path
        if sha256(path)!=feed['sha256']:raise ValueError(f'Cached feed changed: {path}')
        feeds.append(path)
    water=gpd.read_file(BASE/city/'demographics/water.geojson').geometry.union_all()
    return feeds,previous['parameters']['reference_date'],{'water_geometry':water}


def route_city(city,force=False):
    nodes=prepare_nodes(city)
    feeds,reference,geometry=travel_inputs(city)
    parameters=TransitParameters(reference_date=reference,access_policy='uncapped',pedestrian_barrier_policy='reject')
    return compute_city(nodes,feeds,OUTPUT/city/'travel',parameters=parameters,force=force,**geometry)


def analyze_city(city):
    folder=OUTPUT/city
    nodes=pd.read_csv(folder/'nodes.csv',dtype={'tract_geoid':str})
    with np.load(folder/'travel/modes.npz',allow_pickle=False) as saved:modes={key:saved[key] for key in saved.files}
    np.testing.assert_array_equal(modes['tract_geoid'],nodes.tract_geoid)
    travel=json.loads((folder/'travel/manifest.json').read_text())
    new_costs=paired_costs(modes)
    reports=[];summaries=[]
    n=len(nodes);nonlocal_mask=~np.eye(n,dtype=bool)
    old_modes=None
    if city in ORIGINAL_CITIES:
        with np.load(BASE/city/'travel/modes.npz',allow_pickle=False) as saved:old_modes={key:saved[key] for key in saved.files}
        np.testing.assert_allclose(modes['walk_seconds'],old_modes['walk_seconds'],atol=1e-10)
        finite=np.isfinite(old_modes['transit_seconds'])
        if not np.all(modes['transit_seconds'][finite]<=old_modes['transit_seconds'][finite]+1e-6):
            raise ValueError(f'Uncapped transit worsened a previously feasible candidate: {city}')
    for scenario,costs in new_costs.items():
        print(f'  Analyzing {city}: {scenario}',flush=True)
        summary,tracts,arrays=analyze_arrays(nodes,costs)
        # Keep the old same-P majority adaptation as explicitly legacy diagnostics.
        legacy=summary[['group',*[c for c in summary if c.startswith(('fryer_','majority_'))]]].copy()
        summary=summary.drop(columns=[c for c in summary if c.startswith(('fryer_','majority_'))])
        tracts=tracts.drop(columns=[c for c in tracts if c.startswith('fryer_')])
        for frame in (summary,tracts,legacy):
            frame.insert(0,'city',CITIES[city]);frame.insert(0,'city_id',city);frame.insert(0,'scenario',scenario)
            frame.insert(3,'node_scale','70 custom sections' if city=='new_york_city' else 'populated city-limit tract portions')
        summary.insert(4,'node_count',n);summary.insert(5,'city_population',int(nodes.population.sum()))
        out=folder/scenario;out.mkdir(parents=True,exist_ok=True)
        summary.to_csv(out/'segregation_comparison.csv',index=False)
        tracts.to_csv(out/'tract_scores.csv',index=False)
        legacy.to_csv(out/'legacy_travel_majority_diagnostics.csv',index=False)
        np.savez_compressed(out/'analysis_matrices.npz',**arrays)
        np.savez_compressed(out/'cost_matrices.npz',**costs,tract_geoid=nodes.tract_geoid.to_numpy(dtype=str))
        changed=costs['chosen_mode']=='TRANSIT'
        flags=costs['chosen_walk_barrier_flag']
        diagnostics={'city_id':city,'city':CITIES[city],'scenario':scenario,'node_count':n,
            'population':int(nodes.population.sum()),'transit_selected_pairs':int(changed.sum()),
            'nonlocal_pairs':int(nonlocal_mask.sum()),'transit_selected_percent':float(100*changed.sum()/nonlocal_mask.sum()),
            'nodes_without_boarding_access':int((modes['access_stop_count']==0).sum()),
            'old_nodes_without_boarding_access':int((old_modes['access_stop_count']==0).sum()) if old_modes is not None else None,
            'barrier_flagged_direct_walking_pairs':int(flags.sum()),
            'population_weighted_flagged_walking_mass_percent':float(100*np.average((arrays['P']*flags).sum(axis=1),weights=nodes.population)),
            'service_dates':sorted({f['service_date'] for f in travel['feeds']}),
            'bus_routes':sum(f['used_routes_by_mode']['bus'] for f in travel['feeds']),
            'rail_routes':sum(f['used_routes_by_mode']['rail'] for f in travel['feeds']),
            'max_pssi_eigenpair_residual':float(summary.eigenpair_max_abs_residual.max())}
        # Optional detailed walking legs from the revised router.
        for key in ['transit_access_seconds','transit_egress_seconds']:
            if key in modes and changed.any():
                values=modes[key][changed]/60
                diagnostics[key+'_selected_median_minutes']=float(np.median(values))
                diagnostics[key+'_selected_max_minutes']=float(np.max(values))
        report={'city_id':city,'city':CITIES[city],'node_scale':summary.node_scale.iloc[0],
            'scenario':scenario,'parameters':PARAMETERS,'access_policy':'uncapped distance, barrier-screened access and egress; every walking leg charged',
            'walking_transfers':'Automatic transfer links within 400m; explicit GTFS links, transfer restrictions and chained walks retained',
            'no_transit_proxy':True,'diagnostics':diagnostics,
            'source_sha256':{str(path.relative_to(ROOT)):sha256(path) for path in [folder/'nodes.csv',folder/'travel/modes.npz',folder/'travel/manifest.json']},
            'nyc_scope':'NYC uses cached subway/SIR only; no actual bus, PATH, commuter rail, AirTrain or ferry service. Earlier hypothetical NYC transit replacements are removed.' if city=='new_york_city' else None,
            'normalization':'Raw PSSI=lambda; normalized PSSI=(lambda-K)/(1-K). K=all city group residents/all city residents. Per-node right eigenvector sum1; n*v mean1. Distance Fryer is a separate geography-only model.',
            'routes_api_calls':0}
        (out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
        reports.append(diagnostics);summaries.append(summary)
    return pd.concat(summaries,ignore_index=True),reports


def recompute_distance_fryer():
    """Fresh eigenpairs on the unchanged distance graph; no transit can enter."""
    original=ROOT/'outputs/distance_fryer'
    old=pd.read_csv(original/'scores_all_radii.csv')
    old=old[old.radius_km.eq(1)].set_index(['city_id','node_scale','group'])
    rows=[];sources=[]
    for city in CITIES:
        for scale in ['block','tract']:
            folder=original/city/scale
            nodes=pd.read_csv(folder/'nodes.csv',dtype={'node_id':str})
            W=sparse.load_npz(folder/'adjacency_1km.npz')
            for group in FOUR_CATEGORIES:
                stats,_,_=group_ssi(W,nodes[group].to_numpy()/nodes.population.to_numpy(),nodes.dominant_category.eq(group).to_numpy())
                K=float(nodes[group].sum()/nodes.population.sum());value=stats['fryer_ssi']
                previous=old.loc[(city,scale,group)]
                np.testing.assert_allclose(value,previous.fryer_ssi,atol=1e-10,equal_nan=True)
                rows.append({'city_id':city,'city':CITIES[city],'fryer_node_scale':scale,'radius_km':1.,
                    'fryer_node_count':len(nodes),'group':group,'population_share_K':K,
                    'fryer_ssi':value,'fryer_ssi_minus_K':value-K,'fryer_ssi_normalized':normalize_score(value,K),
                    'eligible_assigned_node_count':stats['eligible_assigned_node_count'],'status':stats['status'],
                    'eigenpair_max_abs_residual':stats['max_eigenpair_residual'],
                    'change_from_previous':value-previous.fryer_ssi})
            for path in [folder/'nodes.csv',folder/'adjacency_1km.npz']:
                sources.append({'path':str(path.relative_to(ROOT)),'sha256':sha256(path)})
    result=pd.DataFrame(rows);result.to_csv(OUTPUT/'distance_fryer_recomputed.csv',index=False)
    (OUTPUT/'distance_fryer_sources.json').write_text(json.dumps(sources,indent=2)+'\n')
    return result


def refresh_city_metadata(city):
    """Supplement finished runs without changing any calculated matrices."""
    folder=OUTPUT/city
    with np.load(folder/'travel/modes.npz',allow_pickle=False) as modes:
        access=modes['transit_access_exceeds_legacy_radius']
        egress=modes['transit_egress_exceeds_legacy_radius']
        access_count=modes['access_stop_count']
        for scenario in ['with_transit','walking_only']:
            out=folder/scenario
            with np.load(out/'cost_matrices.npz',allow_pickle=False) as costs:
                chosen=costs['chosen_mode']=='TRANSIT'
            path=out/'manifest.json';report=json.loads(path.read_text())
            report['normalization']='Raw PSSI=lambda; normalized PSSI=(lambda-K)/(1-K). K=all city group residents/all city residents. Per-node right eigenvector sum 1; n*v mean 1. Distance Fryer is a separate geography-only model.'
            report['walking_transfers']='Automatic transfer links within 400m; explicit GTFS links, transfer restrictions and chained walks retained.'
            report['diagnostics'].update({
                'transit_selected_pairs_beyond_old_access_radius':int((chosen&access).sum()),
                'transit_selected_pairs_beyond_old_egress_radius':int((chosen&egress).sum()),
                'transit_selected_pairs_beyond_either_old_radius':int((chosen&(access|egress)).sum()),
                'nodes_without_boarding_access_reason':'No boarding stop passes the retained geometric water/land screen; no access distance cutoff remains.',
            })
            write_json_if_changed(path,report)
        nodes=pd.read_csv(folder/'nodes.csv',dtype={'tract_geoid':str})
        pd.DataFrame({'matrix_index':nodes.matrix_index,'node_id':nodes.tract_geoid,
            'population':nodes.population,'accepted_boarding_stop_count':access_count,
            'no_accepted_boarding_stop':access_count==0}).to_csv(folder/'boarding_access_by_node.csv',index=False)
    path=folder/'travel/manifest.json';report=json.loads(path.read_text())
    report['transfer_walking_interpretation']='Automatic inter-stop links use a 0.4km approximate walking radius and geometry checks; explicit GTFS transfer links and chained walks remain. This is not a total interchange-walk cap.'
    write_json_if_changed(path,report)
    # Refresh manifest hashes after the descriptive travel-manifest correction.
    for scenario in ['with_transit','walking_only']:
        path=folder/scenario/'manifest.json';report=json.loads(path.read_text())
        report['source_sha256'][str((folder/'travel/manifest.json').relative_to(ROOT))]=sha256(folder/'travel/manifest.json')
        write_json_if_changed(path,report)


def aggregate():
    for city in CITIES:refresh_city_metadata(city)
    scores=pd.concat([pd.read_csv(OUTPUT/city/scenario/'segregation_comparison.csv') for city in CITIES for scenario in ['with_transit','walking_only']],ignore_index=True)
    scores.to_csv(OUTPUT/'scenario_scores.csv',index=False)
    tracts=pd.concat([pd.read_csv(OUTPUT/city/scenario/'tract_scores.csv',dtype={'tract_geoid':str}) for city in CITIES for scenario in ['with_transit','walking_only']],ignore_index=True)
    tracts.to_csv(OUTPUT/'all_node_scores.csv',index=False)
    coverage=[]
    for city in CITIES:
        for scenario in ['with_transit','walking_only']:
            coverage.append(json.loads((OUTPUT/city/scenario/'manifest.json').read_text())['diagnostics'])
    coverage=pd.DataFrame(coverage);coverage.to_csv(OUTPUT/'coverage.csv',index=False)
    key=['city_id','city','node_scale','node_count','city_population','group','population','population_share_K']
    paired=scores[scores.scenario.eq('with_transit')][key].copy()
    for scenario in ['with_transit','walking_only']:
        data=scores[scores.scenario.eq(scenario)][['city_id','group',*PSSI_METRICS]].rename(columns={metric:metric+'_'+scenario for metric in PSSI_METRICS})
        paired=paired.merge(data,on=['city_id','group'],validate='one_to_one')
    for metric in PSSI_METRICS:paired[metric+'_walking_minus_transit']=paired[metric+'_walking_only']-paired[metric+'_with_transit']
    fryer=recompute_distance_fryer()
    for scale in ['block','tract']:
        data=fryer[fryer.fryer_node_scale.eq(scale)][['city_id','group','fryer_node_count','fryer_ssi','fryer_ssi_minus_K','fryer_ssi_normalized','status']]
        paired=paired.merge(data.rename(columns={c:f'distance_{scale}_{c}' for c in data if c not in ['city_id','group']}),on=['city_id','group'],validate='one_to_one')
    paired.to_csv(OUTPUT/'city_scores.csv',index=False)
    previous=pd.read_csv(BASE/'transit_comparison/scenario_scores.csv')
    previous=previous[previous.scenario.eq('with_transit')][['city_id','group',*PSSI_METRICS]]
    reference=paired[['city_id','city','group',*[m+'_with_transit' for m in PSSI_METRICS]]].merge(previous,on=['city_id','group'],how='left',validate='one_to_one')
    for metric in PSSI_METRICS:
        reference=reference.rename(columns={metric:metric+'_previous_capped'})
        reference[metric+'_change_new_minus_old']=reference[metric+'_with_transit']-reference[metric+'_previous_capped']
    reference['comparison_scope']=np.where(reference.city_id.eq('new_york_city'),'No comparable capped multicity baseline: NYC now uses actual rail only and the shared expected-time model','Same Census inputs, schedules, economics and water checks; access/egress distance caps removed')
    reference.to_csv(OUTPUT/'previous_results_comparison.csv',index=False)
    manifest={'generated_utc':datetime.now(timezone.utc).isoformat(),'cities':CITIES,'parameters':PARAMETERS,
        'node_policy':'11 cities retain populated city-limit tract portions; NYC retains its 70 custom sections. Do not interpret different spatial scales as identical.',
        'access':'Uncapped walk distance to/from actual stops; full walking time charged. Original conservative water/land chord checks remain. Real bridge/street detours can be missed.',
        'transfer':'Automatic transfer links within 400m remain, along with explicit GTFS links, restrictions and chained walks. This removes access/egress distance caps.',
        'direct_walk':'Same complete great-circle walking proxy in both scenarios: 1.3 distance factor, 4.8 km/h, including flagged possible water crossings.',
        'nyc_change':'Actual cached subway/SIR expected-time service only, no hypothetical transit replacements. Compared with older NYC70 results, routing method/proxy removal also changed.',
        'fryer':'Recomputed 1km dominant-category geographic-neighbor SSI on both blocks and tracts. Transit cannot change it; scores verified against saved prior results. K remains all-city resident fraction. SSI-K and (SSI-K)/(1-K) provided.',
        'source_dates':'Saved service dates retained (Oct 20, 2026 except SF Aug 25, 2026); no fresh schedule downloads.',
        'routes_api_calls':0,'network_requests_during_calculation':0}
    (OUTPUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    write_dictionaries(scores,tracts,paired,fryer)
    write_report(scores,paired,coverage,reference)
    make_plots(scores)
    make_fryer_plot(fryer)
    (OUTPUT/'run_status.json').write_text(json.dumps({'completed_cities':list(CITIES),'city_count':len(CITIES),
        'scenario_category_rows':len(scores),'paired_category_rows':len(paired),'routes_api_calls':0},indent=2)+'\n')
    return scores,paired,coverage


def write_dictionaries(scores,tracts,paired,fryer):
    original=multicity_column_dictionary().set_index('column').definition.to_dict()
    definitions={**original,'scenario':'with_transit: walking + actual GTFS rail/bus with uncapped access/egress. walking_only: identical direct walking estimates and no transit.',
        'node_scale':'Populated city-limit tract portions for 11 cities; 70 custom sections for NYC.',
        'node_count':'Number of populated geographic units in this city PSSI model; dimensions of its P and A matrices.',
        'fryer_node_count':'Number of populated geographic units in the distance-based Fryer model; differs from PSSI for block models and NYC.',
        'fryer_ssi':'Distance-based geographic-neighbor SSI, radius 1 km, dominant category assignment, equal all-neighbor weights, no self-links. Component eigenvalues averaged by eligible assigned-node count; geographic isolates excluded.',
        'fryer_ssi_minus_K':'Distance Fryer SSI minus all-city category population fraction K. Requested adjustment, not simulated null baseline.',
        'fryer_ssi_normalized':'(Distance Fryer SSI - K) / (1 - K). K is all-city category population fraction; negative values retained.',
        'status':'ok or undefined_no_eligible_assigned_nodes. Undefined scores are blank/NaN, not zero.'}
    pd.DataFrame([{'column':c,'definition':definitions[c]} for c in scores]).to_csv(OUTPUT/'score_column_dictionary.csv',index=False)
    tdefs={**tract_column_dictionary().set_index('column').definition.to_dict(),**{k:definitions[k] for k in ['scenario','node_scale']}}
    tdefs['tract_geoid']='Exact node identifier: 11-digit tract GEOID in 11 cities; NYC 70-section zone ID in New York. Load as text; matches matrix ordering.'
    tdefs['population']='2020 population of this city-limit tract portion, or custom section for NYC.'
    tdefs['latitude']='WGS84 representative-point latitude. Eleven cities use the populated-block internal point nearest their population-weighted centroid; NYC retains its saved section representative points.'
    tdefs['longitude']='WGS84 longitude of the same representative point.'
    tdefs['pssi_right_eigenvector']='Node component v_i of the right Perron vector. The vector sums to 1 across each city/category/scenario.'
    tdefs['pssi_eigenvector_times_n']='n*v_i with n the city node count; node mean 1. This is a relative eigenvector score, not a probability, city eigenvalue or scalar normalized PSSI.'
    pd.DataFrame([{'column':c,'definition':tdefs[c]} for c in tracts]).to_csv(OUTPUT/'node_column_dictionary.csv',index=False)
    entries=[]
    for column in paired:
        if column.startswith('distance_'):
            _,scale,field=column.split('_',2);definition=scale.capitalize()+' geographic-node model: '+definitions[field]
        elif column in definitions:definition=definitions[column]
        else:
            definition=None
            for metric in PSSI_METRICS:
                for suffix,label in [('with_transit','Walking + transit, uncapped walking access/egress.'),('walking_only','Identical walking-only geography.'),('walking_minus_transit','Walking-only score minus walking+transit score; absolute score change.')]:
                    if column==metric+'_'+suffix:definition=definitions[metric]+' '+label
            if definition is None:raise ValueError(f'Undefined column:{column}')
        entries.append({'column':column,'definition':definition})
    pd.DataFrame(entries).to_csv(OUTPUT/'city_column_dictionary.csv',index=False)
    fdefs={**definitions,
        'fryer_node_scale':'block: populated 2020 city-limit Census blocks; tract: populated city-limit tract portions. NYC also uses tracts here, unlike its 70-section PSSI model.',
        'radius_km':'Maximum great-circle distance between connected representative points, in kilometers. No self-links.',
        'eligible_assigned_node_count':'Nodes assigned to this category that have at least one neighbor of any category. This is the unweighted city SSI averaging denominator.',
        'eigenpair_max_abs_residual':'Largest absolute component eigenpair residual across this category in this geographic graph.',
        'change_from_previous':'New distance-based Fryer SSI minus the previous 1 km score. Transit does not change the graph, so this should be numerical zero.'}
    pd.DataFrame([{'column':c,'definition':fdefs[c]} for c in fryer]).to_csv(OUTPUT/'fryer_column_dictionary.csv',index=False)


def make_plots(scores):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize,TwoSlopeNorm
    for metric,title in [('segregation_index','PSSI'),('segregation_index_normalized','Normalized PSSI: (λ−K)/(1−K)')]:
        has_negative=scores[metric].min()<0
        norm=TwoSlopeNorm(vmin=-1,vcenter=0,vmax=1) if has_negative else Normalize(0,1)
        cmap='RdBu_r' if has_negative else 'Reds'
        fig,axes=plt.subplots(1,2,figsize=(14,9),layout='constrained')
        for ax,scenario,label in zip(axes,['with_transit','walking_only'],['Walking + transit\nUncapped access/egress','Walking only']):
            frame=scores[scores.scenario.eq(scenario)]
            values=frame.pivot(index='city',columns='group',values=metric).reindex(index=CITIES.values(),columns=FOUR_CATEGORIES).to_numpy()
            image=ax.imshow(values,cmap=cmap,norm=norm,aspect='auto')
            ax.set_xticks(range(4),['White alone','Black alone','Hispanic\n(any race)','Asian alone'],fontsize=10)
            ax.set_yticks(range(12),[name+(' †' if name=='New York City' else '') for name in CITIES.values()])
            ax.set_title(label,fontsize=14);ax.tick_params(length=0)
            for i,j in np.ndindex(values.shape):
                ax.text(j,i,f'{values[i,j]:.3f}',ha='center',va='center',color='white' if abs(values[i,j])>.65 else '#172b3a',fontsize=10)
        fig.colorbar(image,ax=list(axes),shrink=.7)
        fig.suptitle(title+' · revised walking-to-transit model',fontsize=17)
        fig.supxlabel('Full access/egress walking time charged; water checks and existing transfer graph retained.\n† NYC: 70 sections, actual subway/SIR only; other cities: individual tract portions with rail/bus.\nSame direct walking proxy in both scenarios; $15/hour, $3 transit fare, β=0.1, c₀=$1.25.',fontsize=9)
        fig.savefig(OUTPUT/f'{metric}_comparison.png',dpi=190);plt.close(fig)


def make_fryer_plot(fryer):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize,TwoSlopeNorm
    fig,axes=plt.subplots(1,3,figsize=(20,9),layout='constrained')
    frame=fryer[fryer.fryer_node_scale.eq('block')]
    for ax,metric,title in zip(axes,['fryer_ssi','fryer_ssi_minus_K','fryer_ssi_normalized'],['Fryer SSI','Adjusted Fryer: SSI − K','Normalized Fryer: (SSI − K)/(1 − K)']):
        values=frame.pivot(index='city',columns='group',values=metric).reindex(index=CITIES.values(),columns=FOUR_CATEGORIES).to_numpy()
        norm=Normalize(0,1) if metric=='fryer_ssi' else TwoSlopeNorm(vmin=-1,vcenter=0,vmax=1)
        image=ax.imshow(values,cmap='Reds' if metric=='fryer_ssi' else 'RdBu_r',norm=norm,aspect='auto')
        ax.set_xticks(range(4),['White alone','Black alone','Hispanic\n(any race)','Asian alone'],fontsize=9)
        ax.set_yticks(range(12),list(CITIES.values()),fontsize=10)
        ax.set_title(title,fontsize=13);ax.tick_params(length=0)
        for i,j in np.ndindex(values.shape):
            value=values[i,j]
            ax.text(j,i,f'{value:.3f}' if np.isfinite(value) else 'N/A',ha='center',va='center',color='white' if abs(value)>.65 else '#172b3a',fontsize=10)
        fig.colorbar(image,ax=ax,shrink=.65)
    fig.suptitle('Geographic Fryer segregation · 1 km block-neighbor model',fontsize=18)
    fig.supxlabel('Identical with and without transit: this graph contains geographic neighbors only.\nLargest of four categories assigns each block; ties: Hispanic > Asian > Black > White. Geographic isolates excluded from eligible assigned-node averages.\nK = category residents / all city residents. The SSI − K adjustment is the requested population-share adjustment, not a simulated null baseline.',fontsize=10)
    fig.savefig(OUTPUT/'distance_fryer_comparison.png',dpi=180);plt.close(fig)


def write_report(scores,paired,coverage,reference):
    links=''.join(f'<li><a href="maps/{city}/index.html">{escape(name)}: four eigenvector maps</a></li>' for city,name in CITIES.items())
    html='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Revised city segregation: walking to transit</title><style>body{font:16px/1.55 system-ui,sans-serif;background:#f4f7fa;color:#172b3a;margin:30px}main{max-width:1550px;margin:auto}section{background:white;border:1px solid #dce3ea;border-radius:8px;padding:22px;margin:25px 0}img{width:100%;height:auto}.table{overflow:auto;max-height:600px}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:10px;border-bottom:1px solid #dce3ea;white-space:nowrap;text-align:left}th{position:sticky;top:0;background:#eaf0f6}.note{padding:18px;background:#fff1ce;border-left:4px solid #b47d26}a{color:#165d91}p{max-width:1200px}.dictionary td{white-space:normal;min-width:230px}.dictionary td:last-child{min-width:480px}</style></head><body><main><h1>City segregation with walking access to transit</h1>
<p>12 cities · four categories · locally recomputed walking + transit and walking-only scenarios · no Routes API calls.</p>
<div class="note">The previous 1 km bus and 3 km rail access/egress distance caps are removed. Every walking leg still contributes time and cost. Actual GTFS rides are required; there are no hypothetical transit replacements. Conservative water/land chord checks remain and can exclude valid bridge or street detours. Automatic transfer links within 400m, explicit GTFS links, restrictions and chained walks remain.</div>
<p>Populations, geographic units and economic parameters are fixed. Eleven cities retain individual populated 2020 city-limit tract portions; NYC retains 70 custom sections. NYC uses only cached subway/SIR service, with no bus/ferry/commuter rail and no old transit proxies. Its new values are not an isolated access-cap experiment relative to the earlier NYC router.</p>
<p>Walking is great-circle distance × 1.3 at 4.8 km/h. Both scenarios share the same direct-walking approximation, including flagged potential water crossings. Parameters: $15/hour value of time, $3 standardized transit fare, β=0.1 per dollar, c₀=$1.25. Saved schedules: Oct 20, 2026 except San Francisco Aug 25, 2026. These are model estimates, not validated itineraries or observed social interactions.</p>
<p><a href="city_scores.csv">All city PSSI and distance-Fryer scores</a> · <a href="city_column_dictionary.csv">Every city-table column defined</a> · <a href="all_node_scores.csv">All node eigenvectors</a> · <a href="node_column_dictionary.csv">Node column definitions</a> · <a href="previous_results_comparison.csv">Change from previous access-limited results</a> · <a href="verification.json">Verification</a></p>
<section><h2>Updated PSSI comparison</h2><img src="segregation_index_normalized_comparison.png" alt="Normalized PSSI walking-transit versus walking-only"><img src="segregation_index_comparison.png" alt="Raw PSSI comparison"></section>
<section><h2>Four eigenvector maps for every city</h2><p>Maps show n×vᵢ, where each race’s right eigenvector sums to 1 and the mean node value is 1. They are relative node scores, not city eigenvalues or scalar normalized PSSI. All cities use the same displayed logarithmic color range; values outside that range are explicitly flagged and preserved in CSV.</p><p><a href="maps/index.html">Open the complete map gallery</a></p><ul>'''+links+'</ul></section>'
    html+='<section><h2>All segregation scores</h2><img src="distance_fryer_comparison.png" alt="Raw, adjusted and normalized geographic Fryer scores"><p>Distance-based Fryer is independently recomputed on 1 km block and tract graphs. It stays unchanged when transit changes. Columns include raw SSI, SSI−K, and (SSI−K)/(1−K); K is the city’s category resident fraction. This is the requested population-share adjustment, not the paper’s simulated null baseline. <a href="distance_fryer_recomputed.csv">Fresh Fryer calculations</a> · <a href="fryer_column_dictionary.csv">Fryer column definitions</a> · <a href="../../distance_fryer/index.html">Geographic method details and sensitivity</a></p><div class="table">'+paired.to_html(index=False,border=0,float_format=lambda x:f'{x:.6f}',na_rep='N/A')+'</div></section>'
    html+='<section><h2>Access coverage and travel assumptions</h2><div class="table">'+coverage.to_html(index=False,border=0,float_format=lambda x:f'{x:.3f}')+'</div><p><a href="coverage.csv">Coverage CSV</a> · <a href="manifest.json">Full method manifest</a></p></section>'
    html+='<section><h2>Change from the previous access-limited model</h2><div class="table">'+reference.to_html(index=False,border=0,float_format=lambda x:f'{x:.6f}',na_rep='Not comparable')+'</div><p>Removing access restrictions cannot increase any pair’s minimum cost, but the direction of a segregation-score change is not guaranteed because interaction rows are normalized.</p></section>'
    definitions=pd.read_csv(OUTPUT/'city_column_dictionary.csv')
    html+='<section><h2>City table column definitions</h2><div class="table dictionary">'+definitions.to_html(index=False,border=0)+'</div></section>'
    attribution=(BASE/'data_attribution.txt').read_text()
    html+='<section><h2>Dataset attribution</h2>'+''.join('<p>'+escape(part)+'</p>' for part in attribution.split('\n\n'))+'</section></main></body></html>'
    (OUTPUT/'index.html').write_text(html)


def run(cities=None,force=False,route=True,finalize=True):
    OUTPUT.mkdir(parents=True,exist_ok=True)
    completed=[];start=time.perf_counter()
    for city in cities or CITIES:
        print('Revising',CITIES[city],flush=True)
        if route:route_city(city,force=force)
        analyze_city(city)
        completed.append(city)
        (OUTPUT/'run_status.json').write_text(json.dumps({'completed_this_run':completed,'elapsed_seconds':time.perf_counter()-start,'routes_api_calls':0},indent=2)+'\n')
    if finalize:aggregate()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cities',nargs='+',choices=CITIES)
    parser.add_argument('--force',action='store_true')
    parser.add_argument('--analysis-only',action='store_true')
    parser.add_argument('--no-finalize',action='store_true')
    parser.add_argument('--finalize-only',action='store_true')
    args=parser.parse_args()
    if args.finalize_only:aggregate()
    else:run(args.cities,args.force,route=not args.analysis_only,finalize=not args.no_finalize)
