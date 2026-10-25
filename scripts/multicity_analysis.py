"""Comparable tract-level spectral analysis from saved Census and GTFS estimates.

This module never makes a network request. Each row is a populated Census tract
portion within a 2020 city/place boundary. Eigenvalues describe an entire
city/category (or a Fryer component); eigenvector components describe tracts.
"""
from pathlib import Path
from html import escape
import argparse
import hashlib
import json
from zipfile import ZipFile

import networkx as nx
import numpy as np
import pandas as pd

from scripts.nyc_routes import population_transition
from scripts.segregation_metrics import FOUR_CATEGORIES, comparison_column_dictionary, segregation_summary

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'outputs/multicity'
CITIES = {
    'philadelphia':'Philadelphia', 'miami':'Miami', 'los_angeles':'Los Angeles',
    'salt_lake_city':'Salt Lake City', 'san_francisco':'San Francisco',
    'washington_dc':'Washington, DC', 'houston':'Houston', 'detroit':'Detroit',
    'chicago':'Chicago', 'phoenix':'Phoenix', 'atlanta':'Atlanta',
}
PARAMETERS = dict(beta_per_usd=.1, omega_usd_per_hour=15., c0_usd=1.25,
                  standardized_transit_fare_usd=3.)


def perron_pair(matrix, *, tolerance=1e-12, max_iterations=20000):
    """Nonnegative right Perron pair without a full dense eigendecomposition.

    Positive starts and a diagonal shift avoid sign/periodicity problems. Check
    both eigenpair residual and change in the sum-one vector. Nearly disconnected
    walking networks can have an extremely small spectral gap; if iteration does
    not converge, use a dense eigensolver and verify the returned Perron pair.
    A zero matrix has a nonunique vector; the uniform convention is returned.
    """
    M = np.asarray(matrix, float)
    if M.ndim != 2 or M.shape[0] != M.shape[1] or not len(M) or not np.isfinite(M).all() or (M < 0).any():
        raise ValueError('Expected a finite nonnegative nonempty square matrix')
    v = np.full(len(M), 1/len(M))
    if not np.any(M):
        return 0., v, 0., 0
    shift = max(float(M.sum(axis=1).max()) * .01, 1e-15)
    for iteration in range(1, max_iterations+1):
        w = M @ v + shift*v
        v_new = w/w.sum()
        if np.max(np.abs(v_new-v)) < tolerance:
            product = M @ v_new
            eigenvalue = float(product.sum())
            residual = float(np.max(np.abs(product-eigenvalue*v_new)))
            if residual < tolerance:
                return eigenvalue, v_new, residual, iteration
        v = v_new
    from scipy.linalg import eig
    eigenvalues,eigenvectors=eig(M,check_finite=False)
    index=int(np.argmax(eigenvalues.real))
    root=eigenvalues[index]
    candidate=eigenvectors[:,index]
    if abs(root.imag)>tolerance or np.max(np.abs(candidate.imag))>tolerance:
        raise ValueError('Dense fallback did not return a real Perron pair')
    # Absolute values remove the arbitrary eigenvector sign. At a repeated or
    # numerically indistinguishable dominant root they also choose a nonnegative
    # representative; the eigenpair residual must still pass the same tolerance.
    v=np.abs(candidate.real)
    v/=v.sum()
    value=float(root.real)
    residual=float(np.max(np.abs(M@v-value*v)))
    if value < 0 or residual > tolerance or abs(value-float(np.abs(eigenvalues).max()))>tolerance:
        raise ValueError('Perron fallback failed verification; no unverified score exported')
    return value,v,residual,max_iterations


def fryer_tract_scores(P, proportions):
    """Majority-tract SSI, component eigenpairs, and paper-scaled node scores.

    For component C, score_i = |C| * lambda_C * v_C[i], with sum(v_C)=1.
    The mean across all majority tracts equals the city Fryer SSI. Non-majority
    tracts have NaN local scores rather than a fabricated zero score.
    """
    P, p = np.asarray(P, float), np.asarray(proportions, float)
    if P.shape != (len(p), len(p)) or not np.isfinite(P).all() or (P < 0).any() or not np.allclose(P.sum(axis=1),1):
        raise ValueError('Invalid population transition')
    if not np.isfinite(p).all() or ((p < 0)|(p > 1)).any():
        raise ValueError('Invalid race proportions')
    if not np.array_equal(P > 0, P.T > 0):
        raise ValueError('Fryer requires reciprocal support; do not silently symmetrize interactions')
    selected = np.flatnonzero(p >= .5)
    n = len(p)
    details = {
        'fryer_is_majority_tract': p >= .5,
        'fryer_component_id': np.full(n,-1,dtype=int),
        'fryer_component_size': np.zeros(n,dtype=int),
        'fryer_component_eigenvalue': np.full(n,np.nan),
        'fryer_component_eigenvector': np.full(n,np.nan),
        'fryer_tract_ssi': np.full(n,np.nan),
        'fryer_eigenpair_residual': np.full(n,np.nan),
    }
    if not len(selected):
        return (0.,0,0),details
    induced = P[np.ix_(selected,selected)]
    components = list(nx.connected_components(nx.from_numpy_array(induced > 0)))
    city_score = 0.
    for number,component in enumerate(components):
        local = sorted(component)
        indices = selected[local]
        value,vector,residual,_ = perron_pair(induced[np.ix_(local,local)])
        city_score += len(local)*value/len(selected)
        details['fryer_component_id'][indices] = number
        details['fryer_component_size'][indices] = len(local)
        details['fryer_component_eigenvalue'][indices] = value
        details['fryer_component_eigenvector'][indices] = vector
        details['fryer_tract_ssi'][indices] = len(local)*value*vector
        details['fryer_eigenpair_residual'][indices] = residual
    if not np.isclose(np.nanmean(details['fryer_tract_ssi']),city_score,atol=1e-10):
        raise ValueError('Fryer local/city aggregation differs')
    return (city_score,len(selected),len(components)),details


def generalized_costs(walk_seconds,transit_seconds, *, parameters=None):
    params = PARAMETERS if parameters is None else parameters
    walk,transit = np.asarray(walk_seconds,float),np.asarray(transit_seconds,float)
    if walk.shape != transit.shape or walk.ndim != 2 or walk.shape[0] != walk.shape[1]:
        raise ValueError('Matching square mode matrices required')
    if np.isnan(walk).any() or np.isnan(transit).any() or (walk<0).any() or (transit<0).any():
        raise ValueError('Use positive infinity for unavailable candidates, never NaN')
    beta,omega,c0,fare = [params[key] for key in PARAMETERS]
    if not np.isfinite([beta,omega,c0,fare]).all() or beta<=0 or c0<=0 or min(omega,fare)<0:
        raise ValueError('Invalid generalized cost parameters')
    walk_cost = np.where(np.isfinite(walk),omega*np.where(np.isfinite(walk),walk,0)/3600,np.inf)
    transit_cost = fare+np.where(np.isfinite(transit),omega*np.where(np.isfinite(transit),transit,0)/3600,np.inf)
    C = np.maximum(c0,np.minimum(walk_cost,transit_cost))
    mode = np.where(walk_cost<=transit_cost,'WALK','TRANSIT').astype('<U12')
    mode[~np.isfinite(C)]='UNMODELED'
    np.fill_diagonal(C,c0);np.fill_diagonal(mode,'LOCAL')
    return {'C_usd':C,'F':np.exp(-beta*C),'log_F':-beta*C,
            'chosen_mode':mode,'walk_cost_usd':walk_cost,'transit_cost_usd':transit_cost}


def analyze_arrays(nodes, costs):
    """Return summary, all tract/category rows, and ordered analysis arrays."""
    D=nodes.population.to_numpy(dtype=float)
    counts=nodes[list(FOUR_CATEGORIES)].to_numpy(dtype=np.int64)
    p=counts/D[:,None]
    if (D<=0).any() or (counts<0).any() or (p>1).any() or not np.isfinite(costs['C_usd']).all():
        raise ValueError('Invalid demographics or incomplete costs; cannot silently fill missing routes')
    P=population_transition(costs['log_F'],D)
    values=[];vectors=[];residuals=[];matrices=[];fryer_results=[];rows=[]
    for r,name in enumerate(FOUR_CATEGORIES):
        A=P*p[:,r][None,:]
        value,v,residual,iterations=perron_pair(A)
        fryer,fryer_details=fryer_tract_scores(P,p[:,r])
        values.append(value);vectors.append(v);residuals.append(residual);matrices.append(A);fryer_results.append(fryer)
        result=nodes[['matrix_index','tract_geoid','latitude','longitude','population']].copy()
        for extra in ['full_tract_population','city_share_of_tract_population']:
            if extra in nodes:result[extra]=nodes[extra]
        result['small_population_under_100']=D<100
        result['group']=name;result['group_population']=counts[:,r];result['group_proportion']=p[:,r]
        result['population_share_K']=counts[:,r].sum()/D.sum()
        result['pssi_city_eigenvalue']=value
        result['pssi_right_eigenvector']=v
        result['pssi_eigenvector_times_n']=len(nodes)*v
        result['pssi_eigenpair_residual']=residual
        result['network_same_group_exposure']=P@p[:,r]
        result['fryer_city_ssi']=fryer[0]
        for key,data in fryer_details.items():result[key]=data
        rows.append(result)
    summary=segregation_summary(P,D,p,np.array(FOUR_CATEGORIES),np.array(values),np.array(residuals),fryer_results=fryer_results)
    arrays={'P':P,'A':np.array(matrices),'group_names':np.array(FOUR_CATEGORIES),
            'eigenvalues':np.array(values),'right_eigenvectors':np.array(vectors).T,
            'counts':counts,'proportions':p,'population':D,
            'tract_geoid':nodes.tract_geoid.to_numpy(dtype=str)}
    return summary,pd.concat(rows,ignore_index=True),arrays


def analyze_city(city_id, modes_path=None):
    folder=BASE/city_id
    nodes=pd.read_csv(folder/'demographics/nodes.csv',dtype={'tract_geoid':str})
    if nodes.matrix_index.tolist()!=list(range(len(nodes))) or not nodes.tract_geoid.is_unique:
        raise ValueError('Tract ordering invalid')
    modes_path=Path(modes_path) if modes_path else folder/'travel/modes.npz'
    with np.load(modes_path,allow_pickle=False) as saved:
        modes={key:saved[key] for key in saved.files}
    if not np.array_equal(modes['tract_geoid'].astype(str),nodes.tract_geoid.to_numpy()):
        raise ValueError('Travel and Census tract order differ')
    # Prefer barrier-screened walking and real GTFS candidates. A distance-only
    # completion is allowed ONLY for pairs with neither modeled candidate. This
    # avoids an unsupported cheap water crossing beating an available train/bus.
    checked = bool(modes.get('walk_barrier_check_available',False))
    walk=modes['walk_barrier_checked_seconds'].copy() if checked else modes['walk_seconds'].copy()
    strict_costs=generalized_costs(walk,modes['transit_seconds'])
    fallback=~np.isfinite(strict_costs['C_usd'])
    walk[fallback]=modes['walk_seconds'][fallback]
    costs=generalized_costs(walk,modes['transit_seconds'])
    costs['distance_walk_fallback']=fallback
    costs['chosen_source']=np.where(costs['chosen_mode']=='TRANSIT','gtfs_expected_transit','barrier_screened_walk' if checked else 'distance_walk').astype('<U32')
    costs['chosen_source'][fallback]='distance_walk_fallback'
    np.fill_diagonal(costs['chosen_source'],'local_model')
    summary,tracts,arrays=analyze_arrays(nodes,costs)
    out=folder/'analysis';out.mkdir(parents=True,exist_ok=True)
    summary.insert(0,'city',CITIES[city_id]);summary.insert(0,'city_id',city_id)
    summary.insert(2,'populated_tract_portions',len(nodes))
    summary.insert(3,'city_population',int(nodes.population.sum()))
    tracts.insert(0,'city',CITIES[city_id]);tracts.insert(0,'city_id',city_id)
    summary.to_csv(out/'segregation_comparison.csv',index=False)
    tracts.to_csv(out/'tract_scores.csv',index=False)
    np.savez_compressed(out/'analysis_matrices.npz',**arrays)
    np.savez_compressed(out/'cost_matrices.npz',**costs,tract_geoid=nodes.tract_geoid.to_numpy(dtype=str))
    np.savez_compressed(out/'barrier_screened_costs.npz',**strict_costs,tract_geoid=nodes.tract_geoid.to_numpy(dtype=str))
    n=len(nodes);nonlocal_mask=~np.eye(n,dtype=bool)
    fallback_probability_by_origin=(arrays['P']*fallback).sum(axis=1)
    pd.DataFrame({'matrix_index':nodes.matrix_index,'tract_geoid':nodes.tract_geoid,
        'population':nodes.population,'fallback_destination_count':fallback.sum(axis=1),
        'transition_probability_on_fallback':fallback_probability_by_origin}).to_csv(out/'fallback_by_tract.csv',index=False)
    report={'city_id':city_id,'city':CITIES[city_id],'parameters':PARAMETERS,
        'fare_interpretation':'Standardized illustrative $3 per transit journey for comparable scenarios, not actual agency-specific or distance-dependent fares. Transfers do not add extra fare in this scenario.',
        'geography':'Populated 2020 Census tract portions inside 2020 Census city/place limits; block-level counts, no area-weighted racial estimates.',
        'tract_count':n,'population':int(nodes.population.sum()),'categories':list(FOUR_CATEGORIES),
        'tract_portions_under_100_people':int((nodes.population<100).sum()),
        'partially_populated_city_tracts':int((nodes.city_share_of_tract_population<1).sum()) if 'city_share_of_tract_population' in nodes else None,
        'nonlocal_pairs':n*(n-1),'transit_candidate_pairs':int((np.isfinite(modes['transit_seconds'])&nonlocal_mask).sum()),
        'transit_selected_pairs':int((costs['chosen_mode']=='TRANSIT').sum()),
        'walk_selected_pairs':int((costs['chosen_mode']=='WALK').sum()),
        'distance_walk_fallback_pairs':int(fallback.sum()),
        'distance_walk_fallback_percent':float(100*fallback.sum()/(n*(n-1))) if n>1 else 0.,
        'population_weighted_fallback_probability_percent':float(100*np.average(fallback_probability_by_origin,weights=nodes.population)),
        'max_origin_fallback_probability_percent':float(100*fallback_probability_by_origin.max()),
        'travel_completion':'Primary costs prefer barrier-screened walking and actual GTFS transit. Only pairs missing both candidates receive explicitly flagged distance-only WALK proxy completion. These can include impossible water crossings; complete-city scores are exploratory. Barrier-screened costs are also exported with infinity for unresolved pairs.',
        'max_eigenpair_residual':float(summary.eigenpair_max_abs_residual.max()),
        'routes_api_calls':0,'network_requests_during_analysis':0,
        'source_sha256':{str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in [modes_path,folder/'demographics/nodes.csv']},
        'score_interpretation':{
            'pssi':'One dominant city eigenvalue per category; per-tract right eigenvector components sum to 1. n*v is also exported for comparing relative tract scores across different tract counts.',
            'fryer':'Majority-tract adaptation on same P. Each majority-component has an eigenvalue and sum-one eigenvector; tract SSI = component_size * component_eigenvalue * eigenvector. Mean across majority tracts equals city SSI. Nonmajority local scores are undefined, not zero.',
            'normalization':'PSSI=(lambda-K)/(1-K); adjusted Fryer=SSI-K; additionally rescaled Fryer=(SSI-K)/(1-K). K is city population fraction. Negative values retained. No simulated Fryer null baseline is claimed.'}}
    travel_meta=json.loads((folder/'travel/manifest.json').read_text())
    report['service_dates']=sorted({feed['service_date'] for feed in travel_meta['feeds']})
    report['active_bus_routes']=sum(feed.get('used_routes_by_mode',{}).get('bus',0) for feed in travel_meta['feeds'])
    report['active_rail_routes']=sum(feed.get('used_routes_by_mode',{}).get('rail',0) for feed in travel_meta['feeds'])
    report['transit_feeds']=[Path(feed['path']).stem for feed in travel_meta['feeds']]
    report['same_date_across_city_feeds']=travel_meta['common_service_date']
    if 'walk_land_supported' in modes:
        report['walk_pairs_flagged_by_barrier_check']=int((~modes['walk_land_supported'].astype(bool)&nonlocal_mask).sum())
        report['selected_walk_pairs_flagged_by_barrier_check']=int(((costs['chosen_mode']=='WALK')&~modes['walk_land_supported'].astype(bool)).sum())
    (out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    return summary,report


def build_all(city_ids=None):
    rows=[];reports=[]
    city_ids=list(city_ids or CITIES)
    for city in city_ids:
        print(f'Analyzing {CITIES[city]}',flush=True)
        summary,report=analyze_city(city);rows.append(summary);reports.append(report)
    combined=pd.concat(rows,ignore_index=True)
    combined.to_csv(BASE/'city_comparison.csv',index=False)
    pd.concat([pd.read_csv(BASE/city/'analysis/tract_scores.csv',dtype={'tract_geoid':str})
               for city in city_ids],ignore_index=True).to_csv(BASE/'all_tract_scores.csv',index=False)
    pd.DataFrame(reports).drop(columns=['parameters','source_sha256','score_interpretation']).to_csv(BASE/'city_coverage.csv',index=False)
    multicity_column_dictionary().to_csv(BASE/'score_column_dictionary.csv',index=False)
    tract_column_dictionary().to_csv(BASE/'tract_score_dictionary.csv',index=False)
    (BASE/'analysis_manifest.json').write_text(json.dumps({'parameters':PARAMETERS,'cities':reports,'routes_api_calls':0},indent=2)+'\n')
    write_report(combined,reports)
    return combined


def multicity_column_dictionary():
    dictionary=comparison_column_dictionary().copy()
    dictionary['definition']=dictionary.definition.str.replace('all 70 sections','all populated city tract portions',regex=False).str.replace('70 sections','populated city tract portions',regex=False).str.replace('sections','tract portions',regex=False).str.replace('section','tract',regex=False).str.replace('NYC','city',regex=False)
    dictionary['label']=dictionary.label.str.replace('sections','tracts',regex=False)
    extra=pd.DataFrame([
        ('city_id','City ID','Stable folder and file identifier.'),
        ('city','City','2020 Census incorporated place/city; Washington uses the District of Columbia place.'),
        ('populated_tract_portions','Populated tract portions','Number of positive-population Census tract portions inside the city. Tracts crossing its boundary contribute only their city-assigned Census blocks.'),
        ('city_population','City population','Total 2020 population in the selected city limits, independently validated against the published place total.')],columns=dictionary.columns)
    return pd.concat([extra,dictionary],ignore_index=True)


def tract_column_dictionary():
    entries=[
        ('city_id','City folder identifier.'),('city','City name.'),
        ('matrix_index','Zero-based row/column position in all this city’s matrices.'),
        ('tract_geoid','11-character 2020 Census tract GEOID. Preserve as text. Counts include only city-assigned blocks of this tract.'),
        ('latitude','Latitude of representative populated-block internal point nearest the population-weighted centroid, in WGS84 degrees.'),
        ('longitude','Longitude of the same representative point, in WGS84 degrees.'),
        ('population','Total population of the tract portion inside the city.'),
        ('full_tract_population','Population of the entire Census tract, including any part outside city limits.'),
        ('city_share_of_tract_population','Population of the city portion divided by full tract population. Values below 1 identify tracts with residents outside city limits.'),
        ('small_population_under_100','Diagnostic flag for city tract portions with fewer than 100 residents. These are retained to conserve city counts; majority-tract measures can be sensitive to them.'),
        ('group','One of White alone, Black alone, Hispanic any race, or Asian alone; groups overlap.'),
        ('group_population','2020 residents of this category within the city tract portion.'),
        ('group_proportion','group_population / population; fraction in [0,1].'),
        ('population_share_K','Citywide population fraction of this category, repeated for interpretation.'),
        ('pssi_city_eigenvalue','Largest eigenvalue λ of the city’s A for this category. Same across tracts; it is not a tract-specific eigenvalue.'),
        ('pssi_right_eigenvector','Tract’s component v_i of the right Perron vector. Sum across city tracts = 1 for each category.'),
        ('pssi_eigenvector_times_n','n*v_i, where n is the city’s tract count. Mean = 1; gives relative scores on a comparable mean scale across different tract counts, not a probability.'),
        ('pssi_eigenpair_residual','Maximum absolute residual of A*v - λ*v; numerical error, not segregation.'),
        ('network_same_group_exposure','(P*p)_i, modeled same-category exposure for this tract.'),
        ('fryer_city_ssi','Citywide majority-tract Fryer SSI, repeated across rows.'),
        ('fryer_is_majority_tract','True when this category’s tract population share is at least 0.5.'),
        ('fryer_component_id','Zero-based connected majority-component ID. -1 means not applicable because this is not a majority tract.'),
        ('fryer_component_size','Number of majority tracts in this connected component. Zero means not applicable for a nonmajority tract.'),
        ('fryer_component_eigenvalue','Perron eigenvalue of the induced interaction matrix for this tract’s majority component; blank for nonmajority tracts.'),
        ('fryer_component_eigenvector','Component-specific right eigenvector entry; sums to 1 within each majority component. Blank for nonmajority tracts.'),
        ('fryer_tract_ssi','|component| * component_eigenvalue * component_eigenvector. Mean over all majority tracts equals city Fryer SSI. May exceed 1 and is not a probability. Blank for nonmajority tracts.'),
        ('fryer_eigenpair_residual','Maximum eigenpair residual within this majority component; blank for nonmajority tracts.'),
    ]
    return pd.DataFrame(entries,columns=['column','definition'])


def write_report(combined,reports):
    dictionary=multicity_column_dictionary()
    labelmap=dict(zip(dictionary.column,dictionary.label))
    labels={'city_id':'City ID','city':'City','populated_tract_portions':'Populated tract portions','city_population':'City population',**labelmap}
    table=combined.drop(columns='city_id').rename(columns=labels).to_html(index=False,border=0,float_format=lambda x:f'{x:.6f}',
        formatters={labelmap['eigenpair_max_abs_residual']:lambda x:f'{x:.2e}'})
    coverage=pd.DataFrame([{key:report[key] for key in ('city','tract_count','population','active_bus_routes','active_rail_routes','service_dates','same_date_across_city_feeds','tract_portions_under_100_people','partially_populated_city_tracts','nonlocal_pairs','transit_candidate_pairs','transit_selected_pairs','walk_selected_pairs','distance_walk_fallback_pairs','distance_walk_fallback_percent','population_weighted_fallback_probability_percent','max_origin_fallback_probability_percent')} for report in reports])
    citylinks=''.join(f'<li><strong>{escape(report["city"])}</strong>: <a href="{report["city_id"]}/analysis/tract_scores.csv">tract scores</a> · <a href="{report["city_id"]}/analysis/analysis_matrices.npz">A matrices and eigenvectors</a> · <a href="{report["city_id"]}/analysis/cost_matrices.npz">cost matrices</a> · <a href="{report["city_id"]}/travel/manifest.json">travel methods</a> · <a href="{report["city_id"]}/demographics/manifest.json">Census sources</a></li>' for report in reports)
    html='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Eleven-city segregation comparison</title>
<style>body{font:15px/1.5 system-ui,sans-serif;background:#f5f7fa;color:#182b40;margin:30px}main{max-width:1700px;margin:auto}h1{font-size:30px}h2{font-size:21px;margin-top:30px}.table{overflow:auto;background:white;border:1px solid #dae1e9;border-radius:8px;max-height:650px}table{border-collapse:collapse;width:100%}th,td{padding:10px 14px;text-align:left;border-bottom:1px solid #dae1e9}th{background:#e9f0f7;position:sticky;top:0}td{font-variant-numeric:tabular-nums}.results td{white-space:nowrap}.results th{min-width:140px}tbody tr:nth-child(even){background:#f8fafc}p{max-width:1150px}.note{padding:18px;background:#fff3d5;border-left:4px solid #bf8120}a{color:#155990}code{word-break:break-word}</style></head><body><main>
<h1>Segregation across eleven US cities</h1><p>2020 city limits and populated Census tract portions · four overlapping race/ethnicity categories · exploratory walking and public-transit cost estimates</p>
<p><strong>New comparison:</strong> <a href="transit_comparison/index.html">PSSI, normalized PSSI, adjusted Fryer and raw Fryer with transit versus walking only</a>. Both scenarios there share identical walking estimates; this earlier report is retained as a separate reference.</p>
<div class="note"><strong>Model estimates, not observed social contacts.</strong> Walking is a distance approximation screened against Census water polygons; public transit uses downloaded route networks and schedule-derived expected times. Pairs with neither candidate receive an explicitly flagged distance-only walking fallback, which can include impossible water crossings; see the per-city fallback counts below. A standardized illustrative $3 transit fare, $15/hour value of time, β=0.1 per dollar and c₀=$1.25 are shared across cities. These are not city-specific fare quotes or calibrated behavioral parameters. Coverage and feed dates must be considered when comparing cities.</div>
<p><strong>Normalized PSSI:</strong> (λ−K)/(1−K). <strong>Adjusted Fryer:</strong> SSI−K. The extra rescaled Fryer column divides by (1−K). K is the city population fraction. Hispanic is any race and overlaps with the race-alone categories.</p>
<p>There is one PSSI eigenvalue per city/category, not one eigenvalue per tract. Tract files give the associated sum-one eigenvector and n times that component. Fryer files also give each majority component’s eigenvalue and eigenvector, and local SSI = |component| × eigenvalue × eigenvector component. Nonmajority tracts have no local Fryer SSI.</p>
<h2>Comparison</h2><p>Scroll horizontally for all score columns. <a href="city_comparison.csv">Results CSV</a> · <a href="score_column_dictionary.csv">Column dictionary</a></p>'''
    html+=f'<div class="table results">{table}</div><h2>Routing coverage</h2><div class="table">{coverage.to_html(index=False,border=0)}</div>'
    most_completed=sorted(reports,key=lambda x:x['distance_walk_fallback_percent'],reverse=True)[:3]
    coverage_text='; '.join(f'{escape(r["city"])}: {r["distance_walk_fallback_percent"]:.1f}% of pairs' for r in most_completed)
    html+=f'<p><strong>Largest distance-completion shares:</strong> {coverage_text}. The population-weighted fallback-probability column averages Σⱼ Pᵢⱼ × fallbackᵢⱼ over origin populations; the maximum-origin column exposes local dependence that a citywide average can hide. These diagnostics are not error bounds for eigenvalues.</p>'
    count=sum(r['tract_count'] for r in reports)*len(FOUR_CATEGORIES)
    html+=f'<h2>Files by city</h2><p><a href="all_tract_scores.csv">All {count:,} tract/category records in one CSV</a></p><ul>{citylinks}</ul><h2>Score column definitions</h2><div class="table">{dictionary.to_html(index=False,border=0)}</div>'
    html+=f'<h2>Tract column definitions</h2><p><a href="tract_score_dictionary.csv">Tract dictionary CSV</a></p><div class="table">{tract_column_dictionary().to_html(index=False,border=0)}</div>'
    catalog=json.loads((BASE/'transit_sources.json').read_text())
    source_rows=[]
    for entry in catalog['cities']:
        agencies='; '.join(f'<a href="{escape(feed["source_url"],quote=True)}">{escape(feed["agency"])}</a>' for feed in entry['feeds'])
        source_rows.append(f'<tr><td>{escape(entry["city_name"])}</td><td>{agencies}</td><td>{escape(" ".join(entry["coverage_notes"]))}</td></tr>')
    html+='<h2>Transit sources and omitted services</h2><p><a href="transit_sources.json">Full source manifest, download URLs, dates and checksums</a>. Sources include primary downloads and explicitly identified public archives of agency-produced GTFS.</p><div class="table"><table><thead><tr><th>City</th><th>Included agencies</th><th>Coverage notes</th></tr></thead><tbody>'+''.join(source_rows)+'</tbody></table></div>'
    with ZipFile(ROOT/'data/multicity/transit/sfmta.zip') as archive:
        license_name=next(name for name in archive.namelist() if 'License_Agreement' in name)
        license_text=archive.read(license_name).decode('utf-8-sig')
        required_notice=license_text.split('12.',1)[1].split('13.',1)[0].split('\n',1)[1].strip()
    attribution='SFMTA required derivative-data notice:\n'+required_notice+'\n\nRoute and departure data provided by permission of UTA.\n\nContains information from City of Phoenix / Valley Metro GTFS, made available under the ODC Attribution License.\n'
    (BASE/'data_attribution.txt').write_text(attribution)
    html+='<h2>Dataset attribution</h2>'+''.join('<p>'+escape(part)+'</p>' for part in attribution.split('\n\n'))
    html+='<p><a href="https://www.phoenixopendata.com/dataset/valley-metro-bus-schedule">Phoenix dataset</a> · <a href="https://opendatacommons.org/licenses/by/1-0/">ODC Attribution License</a> · <a href="data_attribution.txt">Attribution text</a></p>'
    html+='''<p>City population is the sum of populated city-clipped tract portions, aggregated from 2020 Census blocks. Zero-population portions are excluded from matrix normalization but retained in Census validation. Small populated portions remain included. This matters particularly for Houston: its irregular city boundary yields many tiny portions, which receive equal tract weight in the Fryer component average even though P uses population-weighted destinations. These tract-scale results are not directly interchangeable with the earlier 70-section NYC results.</p>
<p>The <a href="https://eml.berkeley.edu/~fechenique/published/echenique-fryer-qje.pdf">Echenique–Fryer paper</a> defines local component scores on pp. 30–31. Its adjusted residential SSI subtracts a simulated baseline; SSI−K here is explicitly the requested population-share adjustment, not a replication of that simulation.</p></main></body></html>'''
    (BASE/'city_comparison.html').write_text(html)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cities',nargs='+',choices=CITIES)
    args=parser.parse_args()
    build_all(args.cities)
