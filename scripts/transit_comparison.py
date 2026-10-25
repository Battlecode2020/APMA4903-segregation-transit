"""Controlled local comparison: fixed walking estimates, transit on versus off.

The earlier report used a transit-dependent walking completion rule. Here both
scenarios use the same complete geographic walking proxy so that removing transit
cannot silently introduce new walking edges. The previous report is preserved.
No downloads, live routing, or network clients are used.
"""
from html import escape
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

from scripts.multicity_analysis import (
    BASE, ROOT, CITIES, PARAMETERS, analyze_arrays, generalized_costs,
    multicity_column_dictionary, tract_column_dictionary,
)
from scripts.segregation_metrics import FOUR_CATEGORIES

OUTPUT=BASE/'transit_comparison'
SCENARIOS={'with_transit':'Walking + transit','walking_only':'Walking only'}
METRICS={
    'segregation_index':('PSSI','λ'),
    'segregation_index_normalized':('Normalized PSSI','(λ − K) / (1 − K)'),
    'fryer_majority_ssi_minus_K':('Adjusted Fryer','SSI − K'),
    'fryer_majority_ssi':('Fryer SSI','SSI'),
}


def paired_costs(modes):
    """Only the transit candidate changes; the walking matrix is identical."""
    walk=np.asarray(modes['walk_seconds'],float)
    transit=np.asarray(modes['transit_seconds'],float)
    if not np.isfinite(walk).all():
        raise ValueError('The shared geographic walking estimate must be complete')
    scenarios={
        'with_transit':generalized_costs(walk,transit),
        'walking_only':generalized_costs(walk,np.full_like(transit,np.inf)),
    }
    on,off=scenarios['with_transit'],scenarios['walking_only']
    if not np.all(on['C_usd']<=off['C_usd']+1e-12):
        raise ValueError('Removing transit reduced cost: walking inputs must match')
    checked=bool(modes.get('walk_barrier_check_available',False))
    flagged=~np.asarray(modes['walk_land_supported'],bool) if checked else np.zeros(walk.shape,bool)
    np.fill_diagonal(flagged,False)
    for costs in scenarios.values():
        costs['walk_barrier_flag']=flagged
        costs['chosen_walk_barrier_flag']=flagged & (costs['chosen_mode']=='WALK')
        source=np.where(costs['chosen_mode']=='TRANSIT','gtfs_expected_transit','shared_distance_walk').astype('<U32')
        source[costs['chosen_walk_barrier_flag']]='barrier_flagged_walk_proxy'
        np.fill_diagonal(source,'local_model')
        costs['chosen_source']=source
    return scenarios


def compare_results(results):
    """One city/category row with both values and signed differences."""
    key=['city_id','city','group']
    shared=['populated_tract_portions','city_population','population','population_share_K','population_percent','majority_zone_count','fryer_status']
    on=results.loc[results.scenario.eq('with_transit')].set_index(key)
    off=results.loc[results.scenario.eq('walking_only')].set_index(key).reindex(on.index)
    for column in shared:
        if not on[column].equals(off[column]):
            raise ValueError(f'Scenario changed demographic property {column}')
    paired=on[shared].copy()
    for metric in METRICS:
        paired[metric+'_with_transit']=on[metric]
        paired[metric+'_walking_only']=off[metric]
        paired[metric+'_change_without_minus_with']=off[metric]-on[metric]
    return paired.reset_index()


def build():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    summaries=[];tract_tables=[];coverage=[];city_sources=[]
    for city_id,city_name in CITIES.items():
        print(f'Comparing {city_name}',flush=True)
        folder=BASE/city_id
        nodes=pd.read_csv(folder/'demographics/nodes.csv',dtype={'tract_geoid':str})
        mode_path=folder/'travel/modes.npz'
        with np.load(mode_path,allow_pickle=False) as saved:
            modes={key:saved[key] for key in saved.files}
        if not np.array_equal(modes['tract_geoid'].astype(str),nodes.tract_geoid):
            raise ValueError('Tract order differs between demographics and travel')
        costs_by_scenario=paired_costs(modes)
        n=len(nodes);pair_count=n*(n-1)
        for scenario,costs in costs_by_scenario.items():
            summary,tracts,arrays=analyze_arrays(nodes,costs)
            summary.insert(0,'city',city_name);summary.insert(0,'city_id',city_id)
            summary.insert(2,'populated_tract_portions',n)
            summary.insert(3,'city_population',int(nodes.population.sum()))
            summary.insert(0,'scenario',scenario)
            tracts.insert(0,'city',city_name);tracts.insert(0,'city_id',city_id)
            tracts.insert(0,'scenario',scenario)
            out=OUTPUT/city_id/scenario;out.mkdir(parents=True,exist_ok=True)
            summary.to_csv(out/'segregation_comparison.csv',index=False)
            tracts.to_csv(out/'tract_scores.csv',index=False)
            np.savez_compressed(out/'cost_matrices.npz',**costs,tract_geoid=nodes.tract_geoid.to_numpy(dtype=str))
            np.savez_compressed(out/'analysis_matrices.npz',**arrays)
            summaries.append(summary);tract_tables.append(tracts)
            flagged=costs['chosen_walk_barrier_flag']
            mass=(arrays['P']*flagged).sum(axis=1)
            coverage.append({'city_id':city_id,'city':city_name,'scenario':scenario,'tract_count':n,
                'nonlocal_pairs':pair_count,'transit_selected_pairs':int((costs['chosen_mode']=='TRANSIT').sum()),
                'walking_selected_pairs':int((costs['chosen_mode']=='WALK').sum()),
                'selected_walk_pairs_flagged_by_water_check':int(flagged.sum()),
                'selected_flagged_walk_percent':100*flagged.sum()/pair_count,
                'population_weighted_flagged_walk_probability_percent':100*np.average(mass,weights=nodes.population),
                'max_origin_flagged_walk_probability_percent':100*mass.max()})
        travel=json.loads((folder/'travel/manifest.json').read_text())
        city_sources.append({'city_id':city_id,'city':city_name,
            'service_dates':sorted({feed['service_date'] for feed in travel['feeds']}),
            'input_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in [mode_path,folder/'demographics/nodes.csv']}})
    results=pd.concat(summaries,ignore_index=True)
    results.to_csv(OUTPUT/'scenario_scores.csv',index=False)
    paired=compare_results(results)
    paired.to_csv(OUTPUT/'paired_comparison.csv',index=False)
    pd.concat(tract_tables,ignore_index=True).to_csv(OUTPUT/'all_tract_scores.csv',index=False)
    coverage=pd.DataFrame(coverage);coverage.to_csv(OUTPUT/'walking_assumptions.csv',index=False)
    # Preserve a labeled audit against the previous, differently completed model.
    previous=pd.read_csv(BASE/'city_comparison.csv')
    reference=paired[['city_id','city','group']].merge(previous[['city_id','group',*METRICS]],on=['city_id','group'],validate='one_to_one')
    for metric in METRICS:
        reference=reference.rename(columns={metric:metric+'_previous_report'})
        reference[metric+'_paired_with_transit']=paired[metric+'_with_transit']
        reference[metric+'_walking_policy_difference']=reference[metric+'_paired_with_transit']-reference[metric+'_previous_report']
    reference.to_csv(OUTPUT/'previous_report_reference.csv',index=False)
    manifest={'parameters':PARAMETERS,'cities':city_sources,'scenarios':SCENARIOS,
        'walking_policy':'Identical complete great-circle distance walking estimates in both scenarios: distance multiplier1.3, speed4.8km/h. Existing water-crossing flags retained for diagnostics. These are not validated street/bridge routes.',
        'change_from_previous_report':'Previous walking fallback was conditional on absence of transit. Both scenarios here instead share all walking estimates, allowing a controlled transit toggle. Paired with-transit numbers can therefore differ from the earlier report; previous_report_reference.csv quantifies this.',
        'transit_policy':'with_transit uses saved actual bus/rail network expected-time candidates; walking_only sets every transit candidate to infinity. Neither scenario changes populations, coordinates, category shares, beta, omega or c0.',
        'difference_sign':'walking_only minus with_transit; positive means the metric is higher after removing transit in this fixed-population model. This is not a causal estimate of a city actually abolishing transit.',
        'metrics':METRICS,'fryer_policy':'Existing majority-tract adaptation on P, unchanged in both scenarios. Adjusted Fryer is SSI-K, not the paper\'s simulated baseline. No-majority-tract SSI=0 by notebook convention; local SSI undefined.',
        'max_eigenpair_residual':float(results.eigenpair_max_abs_residual.max()),
        'network_requests':0,'routes_api_calls':0}
    (OUTPUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    write_dictionaries()
    return results,paired,coverage,manifest


def write_dictionaries():
    scenario_definition='with_transit = shared walking estimates + saved rail/bus candidates; walking_only = the same walking estimates, with all transit candidates removed.'
    score_dictionary=multicity_column_dictionary()
    pd.concat([pd.DataFrame([{'column':'scenario','label':'Travel scenario','definition':scenario_definition}]),score_dictionary],ignore_index=True).to_csv(OUTPUT/'score_column_dictionary.csv',index=False)
    pd.concat([pd.DataFrame([{'column':'scenario','definition':scenario_definition}]),tract_column_dictionary()],ignore_index=True).to_csv(OUTPUT/'tract_score_dictionary.csv',index=False)
    originals=score_dictionary.set_index('column')
    entries=[]
    for column in pd.read_csv(OUTPUT/'paired_comparison.csv',nrows=0).columns:
        if column in originals.index:
            entries.append({'column':column,**originals.loc[column].to_dict()})
        else:
            for metric,(label,formula) in METRICS.items():
                if column==metric+'_with_transit':
                    entries.append(dict(column=column,label=label+' with transit',definition=formula+' using walking + saved bus/rail candidates. '+originals.loc[metric,'definition']))
                elif column==metric+'_walking_only':
                    entries.append(dict(column=column,label=label+' walking only',definition=formula+' using only the shared walking estimates. '+originals.loc[metric,'definition']))
                elif column==metric+'_change_without_minus_with':
                    entries.append(dict(column=column,label='Change in '+label,definition='Walking-only '+label+' minus walking+transit '+label+'. Positive values mean the metric is higher when transit is removed in this fixed-population scenario; an absolute score difference, not a relative percent change.'))
    pd.DataFrame(entries).to_csv(OUTPUT/'paired_column_dictionary.csv',index=False)


def make_plots(results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize,TwoSlopeNorm
    labels=['White alone','Black alone','Hispanic\n(any race)','Asian alone']
    cities=list(CITIES.values())
    matrices={metric:{scenario:results[results.scenario.eq(scenario)].pivot(index='city',columns='group',values=metric).reindex(index=cities,columns=FOUR_CATEGORIES).to_numpy()
                     for scenario in SCENARIOS} for metric in METRICS}
    no_majority=results[results.scenario.eq('with_transit')].pivot(index='city',columns='group',values='majority_zone_count').reindex(index=cities,columns=FOUR_CATEGORIES).to_numpy()==0
    def draw_pair(fig,axes,metric):
        all_values=np.concatenate(list(matrices[metric].values()))
        if metric in ('segregation_index','fryer_majority_ssi'):
            norm=Normalize(0,1);cmap='Reds'
        else:
            bound=max(.01,float(np.abs(all_values).max()))
            norm=TwoSlopeNorm(vmin=-bound,vcenter=0,vmax=bound);cmap='RdBu_r'
        for ax,(scenario,label) in zip(axes,SCENARIOS.items()):
            values=matrices[metric][scenario]
            image=ax.imshow(values,cmap=cmap,norm=norm,aspect='auto')
            ax.set_xticks(range(4),labels,fontsize=10)
            ax.set_yticks(range(len(cities)),cities,fontsize=10)
            ax.set_title(label,fontsize=13,pad=12)
            ax.tick_params(length=0,pad=7)
            for i,j in np.ndindex(values.shape):
                rgba=image.cmap(norm(values[i,j]));brightness=.2126*rgba[0]+.7152*rgba[1]+.0722*rgba[2]
                marker='*' if metric.startswith('fryer_') and no_majority[i,j] else ''
                ax.text(j,i,f'{values[i,j]:.3f}'+marker,ha='center',va='center',fontsize=10,color='white' if brightness<.5 else '#17283b')
        fig.colorbar(image,ax=list(axes),shrink=.65,label=METRICS[metric][1])
    footnote='Same populations and distance-based walking estimates in both scenarios; only transit availability changes.\nWalking may include unverified water crossings. $15/hour, β=0.1, c₀=$1.25; transit fare=$3 when available.\nFryer: * means no majority tracts (SSI=0 by convention; adjusted SSI=−K). Negative adjustments are retained.'
    for metric,(title,formula) in METRICS.items():
        fig,axes=plt.subplots(1,2,figsize=(13.7,8.3),layout='constrained')
        draw_pair(fig,axes,metric)
        fig.suptitle(f'{title}: {formula} · transit available versus walking only',fontsize=16)
        fig.supxlabel(footnote,fontsize=9)
        fig.savefig(OUTPUT/f'{metric}.png',dpi=190)
        plt.close(fig)
    fig,axes=plt.subplots(4,2,figsize=(15.8,26),layout='constrained')
    for row,(metric,(title,formula)) in enumerate(METRICS.items()):
        draw_pair(fig,axes[row],metric)
        for ax,scenario in zip(axes[row],SCENARIOS):
            ax.set_title(f'{title}: {formula}\n{SCENARIOS[scenario]}',fontsize=14,pad=12)
    fig.suptitle('Eleven cities · four metrics · transit available versus walking only',fontsize=19)
    fig.supxlabel(footnote,fontsize=11)
    fig.savefig(OUTPUT/'all_four_metrics.png',dpi=180)
    plt.close(fig)


def write_report(results,paired,coverage,manifest):
    delta=paired['segregation_index_normalized_change_without_minus_with']
    example=paired[paired.city_id.eq('chicago')&paired.group.eq('black_alone')].iloc[0]
    findings=f'In this scenario, removing transit raises normalized PSSI in {int((delta>1e-10).sum())} of {len(delta)} city/category pairs, although some differences are negligible. Chicago Black alone, for example, changes from {example.segregation_index_normalized_with_transit:.3f} to {example.segregation_index_normalized_walking_only:.3f}. The model holds residential populations fixed; these differences are not predictions of how residents would relocate. Small, geographically isolated tract portions can dominate a spectral score. Tract eigenvectors and population flags are available for inspection.'
    blocks=''.join(f'<section><h2>{escape(title)} · {escape(formula)}</h2><a href="{metric}.png"><img src="{metric}.png" alt="{escape(title)} with transit and walking only"></a></section>' for metric,(title,formula) in METRICS.items())
    descriptions=pd.DataFrame([
        ['PSSI','Dominant eigenvalue λ of A=P diag(p).'],
        ['Normalized PSSI','(λ−K)/(1−K), using citywide population fraction K.'],
        ['Fryer SSI','Existing majority-tract SSI: restrict the same population-weighted P to tracts with p≥0.5; average component eigenvalues by majority-tract counts.'],
        ['Adjusted Fryer','Fryer SSI−K, with no division; not a simulated null-baseline correction.'],
        ['Differences','Walking-only score minus walking+transit score. A positive difference means removing transit increases this model score.'],
        ['K','Category population divided by total city population; identical across scenarios. Hispanic includes any race and overlaps race-alone categories.'],
    ],columns=['Metric / column','Definition'])
    html='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Transit versus walking-only segregation</title>
<style>body{font:16px/1.55 system-ui,sans-serif;background:#f4f7fa;color:#182b40;margin:28px}main{max-width:1450px;margin:auto}h1{font-size:30px}h2{font-size:22px}p{max-width:1150px}.note{background:#fff1cd;padding:18px;border-left:4px solid #bd7a14}section{background:white;padding:20px;margin:25px 0;border:1px solid #dae2eb;border-radius:9px}img{width:100%;height:auto}.table{overflow:auto;max-height:600px}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:10px;border-bottom:1px solid #dae2eb;text-align:left}th{background:#e9f0f7;position:sticky;top:0}.values td{white-space:nowrap}a{color:#165c91}</style></head><body><main>
<h1>Segregation with transit and without transit</h1><p>11 cities · 4 categories · 2020 city-limit tract populations · locally computed rail/bus estimates · no Routes API calls.</p>
<p><strong>New analysis:</strong> <a href="../../distance_fryer/index.html">Distance-based Fryer SSI, population-share adjustments, and four NYC eigenvector maps</a>. The Fryer panels below retain the earlier transit-weighted majority-tract adaptation; use the new report for the requested geographic-neighbor method.</p>
<div class="note"><strong>Controlled comparison:</strong> both scenarios use exactly the same distance-based walking estimates. Transit is the only travel option toggled. The earlier report added walking fallbacks only when transit was absent; that would confound this comparison. Both sides are recomputed here, and the earlier values are kept in a separate reference CSV. Walking can include unverified water crossings; these are exploratory model scenarios, not validated pedestrian itineraries or causal predictions.</div>
<p>Shared parameters: walking distance × 1.3 at 4.8 km/h; value of time=$15/hour; β=0.1 per dollar; local floor=$1.25. Walking costs $0 out of pocket. Transit adds the standardized illustrative $3 fare. Populations, representative points, group shares and majority classifications stay fixed. Saved schedules are October 20, 2026, except San Francisco August 25, 2026.</p>
<p>Each metric uses the <strong>same color scale on both sides</strong>. Different metrics have their own scales. Asterisks identify no-majority-tract Fryer scores, which remain zero by convention (adjusted value=−K); this does not establish absence of segregation among those residents.</p>
<p>COMPARISON_FINDINGS</p>
<p><a href="all_four_metrics.png">Combined eight-panel figure</a> · <a href="paired_comparison.csv">Paired values and differences CSV</a> · <a href="scenario_scores.csv">All 88 scenario/category results</a> · <a href="all_tract_scores.csv">All tract scores</a> · <a href="previous_report_reference.csv">Reference to previous report</a></p>'''
    html=html.replace('COMPARISON_FINDINGS',escape(findings))+blocks
    html+='<section><h2>Metric definitions</h2>'+descriptions.to_html(index=False,border=0)+'<p><a href="score_column_dictionary.csv">All score column definitions</a> · <a href="paired_column_dictionary.csv">Every paired comparison column</a> · <a href="tract_score_dictionary.csv">Tract column definitions</a>. For each metric the paired CSV has suffixes _with_transit, _walking_only, and _change_without_minus_with.</p></section>'
    html+='<section><h2>Values and changes</h2><div class="table values">'+paired.to_html(index=False,border=0,float_format=lambda x:f'{x:.6f}')+'</div></section>'
    html+='<section><h2>Walking approximation diagnostics</h2><p>Water flags refer to straight chords intersecting mapped water. They can identify impossible shortcuts, but can also flag journeys feasible using bridges or an indirect route. Flagged probability mass is weighted by origin population; it is not an eigenvalue error bound.</p><div class="table">'+coverage.to_html(index=False,border=0,float_format=lambda x:f'{x:.3f}')+'</div><p><a href="walking_assumptions.csv">Diagnostics CSV</a></p></section>'
    html+='<p>City-specific cost/analysis NPZ matrices and tract CSVs are stored in each city/scenario subfolder. There is one PSSI eigenvalue per city/category and a right eigenvector entry per tract; Fryer component eigenpairs and local SSI are also exported. <a href="verification.json">Matrix and score verification</a> · <a href="manifest.json">Assumptions, checksums and parameter manifest</a> · <a href="../city_comparison.html">Previous analysis, source coverage and agency omissions</a> · <a href="../transit_sources.json">Transit source manifest</a>.</p>'
    attribution=(BASE/'data_attribution.txt').read_text()
    html+='<section><h2>Dataset attribution</h2>'+''.join('<p>'+escape(part)+'</p>' for part in attribution.split('\n\n'))+'</section></main></body></html>'
    (OUTPUT/'index.html').write_text(html)


if __name__=='__main__':
    data=build()
    make_plots(data[0])
    write_report(*data)
