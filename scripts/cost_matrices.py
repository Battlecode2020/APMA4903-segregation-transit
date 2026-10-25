"""WALK/TRANSIT generalized costs and population/race segregation matrices.

All computation is local. Saved MTA schedules are used without calling Routes.
"""
from datetime import date
from pathlib import Path
import hashlib
import json
import math

import numpy as np
import pandas as pd

from scripts.approximate_travel import approximate_matrices
from scripts.nyc_routes import load_zones, population_transition
from scripts.segregation_metrics import select_four_categories, segregation_summary, write_comparison_report
from scripts.subway_travel import FEED, ROOT, load_feed, scan_connections, walking_links

BASE = ROOT/'outputs/nyc_routes'
OUTPUT = BASE/'cost_model'
FARE_SOURCES = ['https://www.mta.info/fares-tolls/subway-bus',
                'https://www.mta.info/fares-tolls/2025-changes',
                'https://www.mta.info/document/20581']


def fare_class(label, exit_platform, platforms):
    """0 = free SIR trip; 1 = ordinary paid trip. No passes/caps/discounts."""
    rides = [leg for leg in label.legs if leg[0]=='ride']
    if not rides:
        raise ValueError('Transit candidate must include a train ride')
    if any(leg[4] != 'SI' for leg in rides):
        return 1
    first = next(leg[1] for leg in label.legs if leg[0]=='board')
    endpoints = {platforms.iloc[first].parent_station,platforms.iloc[exit_platform].parent_station}
    return int(bool(endpoints & {'S30','S31'}))


def mode_inputs(zones, *, recompute=False):
    """Retain the best scheduled transit time for both zero and paid fare classes."""
    OUTPUT.mkdir(parents=True,exist_ok=True)
    cache=OUTPUT/'mode_inputs.npz'
    metadata_path=OUTPUT/'mode_inputs_manifest.json'
    fingerprint=hashlib.sha256(zones.to_json(orient='records',double_precision=15).encode()).hexdigest()
    feed_sha=hashlib.sha256(FEED.read_bytes()).hexdigest()
    if cache.exists() and metadata_path.exists() and not recompute:
        meta=json.loads(metadata_path.read_text())
        if meta['zone_fingerprint']!=fingerprint or meta['feed_sha256']!=feed_sha:
            raise ValueError('Mode cache differs from current groups/feed; recompute explicitly')
        with np.load(cache,allow_pickle=False) as saved:
            return {k:saved[k] for k in saved.files},meta
    start=8*3600
    platforms,connections,transfers,tables,info,used = load_feed(FEED,date(2026,10,20),start,6*3600)
    # No arbitrary 5 km cutoff for WALK: evaluate any land-mask-supported
    # straight connector in NYC. This remains an approximation, not street routing.
    access,walk,geo,land=walking_links(zones,platforms,max_direct_km=1e6)
    n=len(zones)
    free=np.full((n,n),np.inf)
    paid=np.full((n,n),np.inf)
    platform_records=[]
    for i in range(n):
        arrivals=scan_connections(connections,transfers,access[i],start)
        for p,label in enumerate(arrivals):
            if label is None:continue
            fare= fare_class(label,p,platforms)
            elapsed=label.time+access[:,p]-start
            valid=np.isfinite(elapsed) & (elapsed<=6*3600)
            valid[i]=False
            target=paid if fare else free
            improved=valid & (elapsed<target[i])
            target[i,improved]=elapsed[improved]
        if (i+1)%10==0:print(f'Independent WALK/TRANSIT candidates: {i+1}/70 origins',flush=True)
    arrays={'walk_seconds':walk,'transit_free_seconds':free,'transit_paid_seconds':paid,
            'transit_proxy_seconds':approximate_matrices(zones[['latitude','longitude']].to_numpy())['duration_seconds'],
            'great_circle_distance_km':geo,'zone_ids':zones.zone_id.to_numpy(dtype=str)}
    np.savez_compressed(cache,**arrays)
    meta={'zone_fingerprint':fingerprint,'feed_sha256':feed_sha,'feed_info':info,
          'service_date':'2026-10-20','departure_local':'08:00:00','timezone':'America/New_York',
          'fare_sources':FARE_SOURCES,'fare_source_checked_on':'2026-10-19',
          'walking':'4.8 km/h, distance multiplier 1.3, 60 m land buffer; no distance cap for direct WALK',
          'transit':'Saved subway/SIR GTFS; 3 km access/egress limit, 2 min entry/exit, 6 h arrival window, same transfer rules as subway_travel.py',
          'fare_classes':'Paid: standard adult single-trip subway/SIR fare. Free: SIR-only with neither boarding nor exiting at S30/S31.',
          'candidate_limit':'Earliest-arrival labels per platform; best zero/paid fare candidates retained. Not an enumeration of every possible fare-aware transit itinerary.',
          'proxy':'Only when no scheduled transit candidate exists: previous baseline (18 km/h, 1.4 distance multiplier, 10 min overhead), interpreted explicitly as a hypothetical transit proxy.',
          'routes_api_calls':0,'network_requests':0}
    metadata_path.write_text(json.dumps(meta,indent=2)+'\n')
    return arrays,meta


def combine_modes(inputs, *, beta=.1,omega=15.,c0=1.25,transit_fare=3.,use_transit_proxy=True):
    if not all(math.isfinite(x) and x>=0 for x in (beta,omega,c0,transit_fare)) or beta==0 or c0==0:
        raise ValueError('Finite nonnegative parameters and positive beta/c0 required')
    walk,free,paid,proxy=[np.asarray(inputs[k],float) for k in ('walk_seconds','transit_free_seconds','transit_paid_seconds','transit_proxy_seconds')]
    if walk.ndim!=2 or walk.shape[0]!=walk.shape[1] or any(a.shape!=walk.shape for a in (free,paid,proxy)):
        raise ValueError('Mode matrices must have matching square shapes')
    for a in (walk,free,paid,proxy):
        if np.isnan(a).any() or (a<0).any():raise ValueError('Mode times must be nonnegative; use infinity for unmodeled candidates')
    # Avoid 0 * infinity when omega=0: unsupported modes remain unsupported.
    def time_cost(t):
        return np.where(np.isfinite(t),omega*np.where(np.isfinite(t),t,0)/3600,np.inf)
    walk_cost=time_cost(walk)
    free_cost=time_cost(free)
    paid_cost=transit_fare+time_cost(paid)
    use_free=free_cost<=paid_cost
    transit_cost=np.minimum(free_cost,paid_cost)
    transit_time=np.where(use_free,free,paid)
    transit_price=np.where(use_free,0.,transit_fare)
    has_scheduled=np.isfinite(transit_cost)
    proxy_used=(~has_scheduled) & ~np.eye(len(walk),dtype=bool) & use_transit_proxy
    transit_cost=np.where(proxy_used,transit_fare+time_cost(proxy),transit_cost)
    transit_time=np.where(proxy_used,proxy,transit_time)
    transit_price=np.where(proxy_used,transit_fare,transit_price)
    transit_price[~np.isfinite(transit_cost)]=np.nan
    # Deterministic tie rule: WALK; floor is applied only after minimization.
    choose_walk=walk_cost<=transit_cost
    raw=np.minimum(walk_cost,transit_cost)
    cost=np.maximum(c0,raw)
    complete=np.isfinite(raw)
    cost[~complete]=np.nan
    chosen_mode=np.where(choose_walk,'WALK','TRANSIT').astype('<U12')
    chosen_mode[~complete]='UNMODELED'
    chosen_source=np.where(choose_walk,'walk_distance','scheduled_transit').astype('<U32')
    chosen_source[(~choose_walk)&proxy_used]='transit_proxy'
    chosen_source[~complete]='unmodeled'
    chosen_time=np.where(choose_walk,walk,transit_time)
    chosen_price=np.where(choose_walk,0.,transit_price)
    # Local convention is independent of transit fares or access overhead.
    np.fill_diagonal(cost,c0)
    np.fill_diagonal(chosen_time,0)
    np.fill_diagonal(chosen_price,0)
    np.fill_diagonal(chosen_mode,'LOCAL')
    np.fill_diagonal(chosen_source,'local_model')
    log_f=-beta*cost
    return {'C_usd':cost,'F':np.exp(log_f),'log_F':log_f,'chosen_mode':chosen_mode,'chosen_source':chosen_source,
            'selected_seconds':chosen_time,'selected_price_usd':chosen_price,
            'walk_seconds':walk,'transit_seconds':transit_time,'transit_price_usd':transit_price,
            'walk_generalized_cost_usd':walk_cost,'transit_generalized_cost_usd':transit_cost,
            'transit_proxy_candidate':proxy_used,'scheduled_transit_available':has_scheduled}


def race_analysis(log_f,populations,proportions):
    """A[r,i,j] = P[i,j] * proportion[j,r]; normalized RIGHT Perron vectors."""
    populations=np.asarray(populations,float)
    proportions=np.asarray(proportions,float)
    if proportions.ndim!=2 or proportions.shape[0]!=len(populations) or not np.isfinite(proportions).all() or ((proportions<0)|(proportions>1)).any():
        raise ValueError('Need a finite n by r proportion array in [0,1]')
    P=population_transition(log_f,populations)
    A=np.stack([P*proportions[:,r][None,:] for r in range(proportions.shape[1])])
    eigenvalues=[];vectors=[];residuals=[]
    for matrix in A:
        if not np.any(matrix):
            raise ValueError('Zero race matrix has no unique normalized eigenvector')
        values,raw_vectors=np.linalg.eig(matrix)
        k=int(np.argmax(values.real))
        value=values[k]
        vector=raw_vectors[:,k]
        if abs(value.imag)>1e-9 or np.max(abs(vector.imag))>1e-8:
            raise ValueError('Could not identify a real Perron eigenpair')
        vector=vector.real
        if vector.sum()<0:vector=-vector
        if vector.min()<-1e-9 or abs(vector.sum())<1e-14:
            raise ValueError('Dominant vector is not nonnegative')
        vector=np.maximum(vector,0);vector/=vector.sum()
        lam=float(value.real)
        residual=float(np.max(abs(matrix@vector-lam*vector)))
        if not -1e-10<=lam<=1+1e-10 or residual>1e-9:
            raise ValueError('Invalid segregation eigenpair')
        eigenvalues.append(lam);vectors.append(vector);residuals.append(residual)
    return P,A,np.array(eigenvalues),np.array(vectors).T,np.array(residuals)


def build(*,beta=.1,omega=15.,c0=1.25,transit_fare=3.,recompute_modes=False):
    zones=load_zones(BASE/'zones.csv')
    if len(zones)!=70:raise ValueError('Expected 70 groups')
    inputs,mode_meta=mode_inputs(zones,recompute=recompute_modes)
    with np.load(BASE/'demographics_2020/demographics_2020.npz',allow_pickle=False) as data:
        demographics={k:data[k] for k in data.files}
    if not np.array_equal(inputs['zone_ids'],demographics['zone_ids']) or not np.array_equal(demographics['zone_ids'],zones.zone_id):
        raise ValueError('Demographic/route group order differs')
    if not np.array_equal(demographics['population'],zones.TOTPOP):raise ValueError('Population mismatch')
    four_names,four_counts,four_proportions=select_four_categories(demographics)
    demographics.update(four_category_names=four_names,four_category_counts=four_counts,
                        four_category_proportions=four_proportions)
    ids=zones.zone_id.to_numpy(dtype=str)
    report={'parameters':{'beta_per_usd':beta,'omega_usd_per_hour':omega,'c0_usd':c0,'standard_transit_fare_usd':transit_fare},
            'modes':['WALK','TRANSIT'],'groups':70,'matrix_shape':[70,70],
            'definition':'C=max(c0,min_m(price_m+omega*time_hours_m)); F=exp(-beta*C); A_r[i,j]=D[j]*F[i,j]*p[j,r]/sum_k(F[i,k]*D[k])',
            'source':mode_meta,'variants':{},'routes_api_calls':0,
            'primary_categories':four_names.tolist(),
            'comparison_methods':{
                'population_share_K':'sum_i(D_i*p_ir)/sum_i(D_i); 2020 NYC total population is the denominator for every category',
                'overlap':'Hispanic means any race. White, Black and Asian alone include Hispanic residents; the four categories are not a partition.',
                'normalization':'(score-K)/(1-K), using K as a fraction. Negative values are retained; K=1 is undefined (NaN). This is a population-share adjustment, not a simulated chance correction.',
                'fryer_population_adjustment':'fryer_majority_ssi_minus_K = SSI-K, without division. The older (SSI-K)/(1-K) column remains a separate rescaled comparison.',
                'paper_baseline_adjustment':'The paper pp. 24-25 subtracts a simulated random-assignment baseline, not generally K. That simulated baseline is not computed here; SSI-K is explicitly a population-share adjustment.',
                'fryer_majority_ssi':'70-zone adaptation of notebook Fryer_SSI: select p_ir>=0.5, restrict the SAME population-normalized P used by PSSI, retain diagonal and original weights (no induced-row renormalization), average component spectral radii weighted by component zone counts. Zero when no majority zones, by notebook convention.',
                'fryer_limit':'Adaptation to modeled travel interactions and large heterogeneous zones; not the original paper\'s 1-km block-neighbor empirical network or an individual-level estimate. K is population share, not majority-zone share, as requested.',
                'fryer_source':'https://eml.berkeley.edu/~fechenique/published/echenique-fryer-qje.pdf',
                'network_isolation':'sum_i[(D_i*p_ir / sum_j D_j*p_jr) * (P @ p_r)_i]',
                'residential_isolation':'sum_i[(D_i*p_ir / sum_j D_j*p_jr) * p_ir]',
                'dissimilarity_vs_rest':'0.5 * sum_i abs(D_i*p_ir/N_r - D_i*(1-p_ir)/(N-N_r)); undefined for absent or universal categories',
                'section_scores':'Right Perron vectors sum to 1; scalar population-share normalization is not applied to these vector components.'},
            'limitations':['Modeled candidates only; missing walk links or transit routes are not proven impossible.',
                'Complete estimate uses an uncalibrated transit proxy only where scheduled rail candidates are absent; proxy fare is an assumed standard fare, not a quoted itinerary price.',
                'Public transit coverage is subway/SIR, with the proxy standing in for missing services. No bus/ferry network is implemented.',
                '2020 population and race counts; stated 2026 schedule and fares.',
                'Standard adult single trip; no fare caps, passes, discounts, premium services or rider fare history.',
                'beta, omega and c0 remain illustrative; right eigenvector components are normalized relative model scores, not local probabilities of segregation.']}
    for variant,use_proxy in [('estimated',True),('scheduled_candidates_only',False)]:
        arrays=combine_modes(inputs,beta=beta,omega=omega,c0=c0,transit_fare=transit_fare,use_transit_proxy=use_proxy)
        folder=OUTPUT/variant;folder.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(folder/'cost_matrices.npz',**arrays,zone_ids=ids,population=zones.TOTPOP.to_numpy())
        for name in arrays:
            pd.DataFrame(arrays[name],index=ids,columns=ids).rename_axis('origin_zone').to_csv(folder/f'{name}.csv')
        report['variants'][variant]={'unresolved_cells':int(np.isnan(arrays['C_usd']).sum()),
            'walk_selected':int((arrays['chosen_mode']=='WALK').sum()),'transit_selected':int((arrays['chosen_mode']=='TRANSIT').sum()),
            'selected_transit_proxy':int((arrays['chosen_source']=='transit_proxy').sum()),
            'proxy_candidates':int(arrays['transit_proxy_candidate'].sum())}
        if np.isnan(arrays['C_usd']).any():continue
        for category,p_key,names_key in [('four_category','four_category_proportions','four_category_names'),('race','race_proportions','race_names'),('race_ethnicity','race_ethnicity_proportions','race_ethnicity_names')]:
            names=demographics[names_key]
            proportions=demographics[p_key]
            P,A,lam,v,residual=race_analysis(arrays['log_F'],demographics['population'],proportions)
            np.savez_compressed(folder/f'{category}_analysis.npz',P=P,A=A,eigenvalues=lam,right_eigenvectors=v,
                                proportions=proportions,counts=demographics[f'{category}_counts'],
                                population=demographics['population'],group_names=names,zone_ids=ids)
            comparison=segregation_summary(P,demographics['population'],proportions,names,lam,residual)
            comparison.to_csv(folder/f'{category}_indices.csv',index=False)
            vectors=pd.DataFrame(v,columns=names)
            vectors.insert(0,'zone_id',ids);vectors.insert(0,'matrix_index',np.arange(len(ids)))
            vectors.to_csv(folder/f'{category}_section_scores.csv',index=False)
            if category=='four_category':
                write_comparison_report(comparison,folder,parameters=report['parameters'],
                                        total_population=int(demographics['population'].sum()),zone_count=len(zones))
                section_rows=[]
                for r,name in enumerate(names):
                    section_rows.append(pd.DataFrame({'matrix_index':np.arange(len(ids)),'zone_id':ids,
                        'borough':zones.borough.to_numpy(),'group':name,'population':demographics['population'],
                        'group_population':four_counts[:,r],'group_proportion':proportions[:,r],
                        'is_majority_zone':proportions[:,r]>=.5,'network_same_group_exposure':P@proportions[:,r],
                        'right_eigenvector_score':v[:,r]}))
                pd.concat(section_rows,ignore_index=True).to_csv(folder/'four_category_section_diagnostics.csv',index=False)
            for r,name in enumerate(names):
                pd.DataFrame(A[r],index=ids,columns=ids).rename_axis('origin_zone').to_csv(folder/f'A_{name}.csv')
            pd.DataFrame(P,index=ids,columns=ids).rename_axis('origin_zone').to_csv(folder/'P_population_transition.csv')
            # Independently verify the user's literal formula, including diagonal and D(j).
            F=arrays['F'];D=demographics['population']
            explicit=(F[None,:,:]*D[None,None,:]*proportions.T[:,None,:])/(F@D)[None,:,None]
            if not np.allclose(explicit,A,rtol=1e-11,atol=1e-13):raise ValueError('Literal A formula mismatch')
            if not np.allclose(P.sum(axis=1),1) or not np.allclose(v.sum(axis=0),1):raise ValueError('Normalization check failed')
    zones[['matrix_index','zone_id','borough','TOTPOP','latitude','longitude']].to_csv(OUTPUT/'zone_order.csv',index=False)
    (OUTPUT/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    report=build()
    print(json.dumps({'parameters':report['parameters'],'variants':report['variants']},indent=2))
