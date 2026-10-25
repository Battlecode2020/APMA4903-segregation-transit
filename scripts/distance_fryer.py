"""Distance-neighbor Echenique–Fryer SSI using saved 2020 Census populations.

Main analysis: populated Census blocks; companion: populated city-limit tracts.
Neighbors are distinct representative points within a great-circle radius. Equal
neighbor intensities, no population/travel weighting, no self links. Degree-zero
nodes are excluded from the group denominator as in the authors' neighbors.m.
No network operations or Routes API calls occur in this module.
"""
from html import escape
from pathlib import Path
from zipfile import ZipFile
import hashlib
import json

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.linalg import eigh
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh
from scipy.spatial import cKDTree

from scripts.multicity_analysis import ROOT, BASE, CITIES as ELEVEN_CITIES
from scripts.multicity_census import geo_layout, VARIABLES
from scripts.segregation_metrics import FOUR_CATEGORIES, normalize_score

OUTPUT=ROOT/'outputs/distance_fryer'
CITIES={**ELEVEN_CITIES,'new_york_city':'New York City'}
RADII_KM=(.5,1.,1.5)
PRIMARY_RADIUS_KM=1.
EARTH_RADIUS_KM=6378.1  # Authors' neighbors.m convention.
COUNTS=['population',*FOUR_CATEGORIES]
PAPER_URL='https://eml.berkeley.edu/~fechenique/published/echenique-fryer-qje.pdf'
LABELS=['White alone','Black alone','Hispanic\n(any race)','Asian alone']


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda:source.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def nyc_blocks():
    """Read NYC city-assigned blocks from the previously downloaded P.L. archive."""
    cached=OUTPUT/'inputs/new_york_city_blocks.csv.gz'
    if cached.exists():return pd.read_csv(cached,dtype={'node_id':str})
    with ZipFile(ROOT/'data/supplements/ny2020.pl.zip') as archive:
        geography=[]
        with archive.open('nygeo2020.pl') as stream:
            for chunk in pd.read_csv(stream,sep='|',header=None,names=geo_layout(),
                usecols=['LOGRECNO','SUMLEV','GEOVAR','GEOCOMP','CHARITER','GEOCODE','PLACE','INTPTLAT','INTPTLON','POP100'],
                dtype=str,keep_default_na=False,chunksize=100000,encoding='latin-1'):
                keep=chunk.PLACE.eq('51000') & chunk.SUMLEV.isin(['160','750']) & chunk.GEOVAR.eq('00') & chunk.GEOCOMP.eq('00') & chunk.CHARITER.eq('000')
                geography.append(chunk[keep])
        geo=pd.concat(geography,ignore_index=True);ids=set(geo.LOGRECNO)
        names=['FILEID','STUSAB','CHARITER','CIFSN','LOGRECNO']
        names += [f'P001{i:04d}' for i in range(1,72)]+[f'P002{i:04d}' for i in range(1,74)]
        data=[]
        with archive.open('ny000012020.pl') as stream:
            for chunk in pd.read_csv(stream,sep='|',header=None,names=names,usecols=['LOGRECNO',*VARIABLES.values()],dtype={'LOGRECNO':str},chunksize=100000):
                data.append(chunk[chunk.LOGRECNO.isin(ids)])
    records=geo.merge(pd.concat(data),on='LOGRECNO',validate='one_to_one').rename(columns={v:k for k,v in VARIABLES.items()})
    assert records[COUNTS].notna().all().all()
    np.testing.assert_array_equal(records.population,records.POP100.astype(int))
    official=records[records.SUMLEV.eq('160')]
    assert len(official)==1 and official.population.iloc[0]==8804190
    blocks=records[records.SUMLEV.eq('750')].copy()
    np.testing.assert_array_equal(blocks[COUNTS].sum(),official[COUNTS].iloc[0])
    blocks=blocks.rename(columns={'GEOCODE':'node_id','INTPTLAT':'latitude','INTPTLON':'longitude'})[['node_id','latitude','longitude',*COUNTS]]
    blocks[['latitude','longitude']]=blocks[['latitude','longitude']].astype(float)
    blocks[COUNTS]=blocks[COUNTS].astype(np.int64)
    cached.parent.mkdir(parents=True,exist_ok=True)
    blocks.to_csv(cached,index=False,compression='gzip')
    return blocks


def load_nodes(city,scale):
    if scale=='block':
        if city=='new_york_city':
            nodes=nyc_blocks()
            source=OUTPUT/'inputs/new_york_city_blocks.csv.gz'
        else:
            source=BASE/city/'demographics/city_blocks.csv.gz'
            nodes=pd.read_csv(source,dtype={'GEOCODE':str}).rename(columns={'GEOCODE':'node_id','INTPTLAT':'latitude','INTPTLON':'longitude'})
    elif scale=='tract':
        if city=='new_york_city':
            source=ROOT/'outputs/ny_2020/tracts/nodes.csv'
            raw=pd.read_csv(source,dtype={'GEOID20':str,'COUNTYFP20':str})
            raw=raw[raw.COUNTYFP20.isin(['005','047','061','081','085'])]
            nodes=pd.DataFrame({'node_id':raw.GEOID20,'latitude':raw.INTPTLAT20,'longitude':raw.INTPTLON20,
                'population':raw.TOTPOP,'white_alone':raw.NH_WHITE+raw.H_WHITE,
                'black_alone':raw.NH_BLACK+raw.H_BLACK,'hispanic_latino_any_race':raw.HISP,'asian_alone':raw.NH_ASIAN+raw.H_ASIAN})
        else:
            source=BASE/city/'demographics/nodes.csv'
            nodes=pd.read_csv(source,dtype={'tract_geoid':str}).rename(columns={'tract_geoid':'node_id'})
    else:raise ValueError('Expected block or tract scale')
    zero=int((nodes.population==0).sum())
    nodes=nodes[nodes.population>0][['node_id','latitude','longitude',*COUNTS]].sort_values('node_id').reset_index(drop=True)
    nodes.insert(0,'matrix_index',np.arange(len(nodes)))
    assert nodes.node_id.is_unique and np.isfinite(nodes[['latitude','longitude']]).all().all()
    assert (nodes[COUNTS]>=0).all().all() and (nodes[list(FOUR_CATEGORIES)].to_numpy()<=nodes.population.to_numpy()[:,None]).all()
    if city!='new_york_city':
        published=json.loads((BASE/city/'demographics/manifest.json').read_text())
        expected=[published['total_population'],*[published['race_counts'][r] for r in FOUR_CATEGORIES]]
    else:
        expected=[8804190,3000945,1943645,2490350,1385144]
    np.testing.assert_array_equal(nodes[COUNTS].sum(),expected)
    metadata={'city_id':city,'city':CITIES[city],'node_scale':scale,'node_count':len(nodes),'zero_population_nodes_excluded':zero,
        'city_population':int(nodes.population.sum()),'source':str(source.relative_to(ROOT)),'source_sha256':sha256(source),
        'representative_points':'Census populated-block internal points' if scale=='block' else ('Census tract internal points' if city=='new_york_city' else 'Existing populated-block point nearest the city-portion population centroid')}
    return nodes,metadata


def neighbor_graph(latitude,longitude,radius_km):
    """Symmetric sparse binary graph of distinct nodes, inclusive cutoff."""
    lat,lon=np.radians(latitude),np.radians(longitude)
    if not np.isfinite(radius_km) or radius_km<=0 or radius_km>=np.pi*EARTH_RADIUS_KM:
        raise ValueError('Invalid neighbor radius')
    if lat.ndim!=1 or lon.shape!=lat.shape or not np.isfinite(lat).all() or not np.isfinite(lon).all():
        raise ValueError('Invalid coordinate vectors')
    xyz=np.column_stack((np.cos(lat)*np.cos(lon),np.cos(lat)*np.sin(lon),np.sin(lat)))
    chord=2*np.sin(radius_km/(2*EARTH_RADIUS_KM))
    pairs=cKDTree(xyz).query_pairs(chord,output_type='ndarray')
    row=np.concatenate((pairs[:,0],pairs[:,1]));col=np.concatenate((pairs[:,1],pairs[:,0]))
    return sparse.csr_matrix((np.ones(len(row),dtype=np.uint8),(row,col)),shape=(len(lat),len(lat)))


def component_pair(adjacency,degrees):
    """Perron pair of B=diag(1/all-neighbor-degree)*same-group adjacency.

    Use its symmetric similar matrix to handle bipartite/slow-mixing components.
    The selected eigenvalue is largest algebraic, not largest absolute negative.
    """
    n=len(degrees)
    if not adjacency.nnz:return 0.,np.full(n,1/n),0.
    root_degree=np.sqrt(degrees)
    inverse=sparse.diags(1/root_degree)
    symmetric=inverse@adjacency@inverse
    if n<=48:
        values,vectors=eigh(symmetric.toarray(),subset_by_index=[n-1,n-1])
    else:
        values,vectors=eigsh(symmetric,k=1,which='LA',v0=np.ones(n)/np.sqrt(n),tol=1e-11,maxiter=max(10000,10*n))
    value=float(values[-1]);v=vectors[:,-1]/root_degree
    if v.sum()<0:v=-v
    if v.min() < -1e-8:raise ValueError('Non-Perron sign pattern')
    v=np.maximum(v,0);v/=v.sum()
    residual=float(np.max(np.abs(adjacency@v/degrees-value*v)))
    if residual>1e-10 or value < -1e-12 or value > 1+1e-10:raise ValueError('Unverified Fryer eigenpair')
    return float(np.clip(value,0,1)),v,residual


def dominant_labels(counts):
    """Largest of four counts; ties Hispanic > Asian > Black > White.

    This explicit tie convention matches the authors' published sample; their
    original preprocessing code is not available. All-four-zero nodes stay other.
    """
    counts=np.asarray(counts)
    if counts.ndim!=2 or counts.shape[1]!=4 or not np.isfinite(counts).all() or (counts<0).any():
        raise ValueError('Expected four nonnegative category count columns')
    priority=np.array([2,3,1,0])  # FOUR_CATEGORIES order: white, black, Hispanic, Asian.
    labels=priority[np.argmax(counts[:,priority],axis=1)]
    labels[counts.max(axis=1)==0]=-1
    return labels


def group_ssi(adjacency,proportions,assigned):
    """Category-induced network; outside-group nodes stay in neighbor degrees."""
    p=np.asarray(proportions,float);W=sparse.csr_matrix(adjacency)
    if W.shape!=(len(p),len(p)) or (W!=W.T).nnz or np.any(W.diagonal()) or (W.data!=1).any():
        raise ValueError('Expected symmetric binary neighbor matrix without self-links')
    if not np.isfinite(p).all() or ((p<0)|(p>1)).any():raise ValueError('Invalid proportions')
    degrees=np.asarray(W.sum(axis=1)).ravel()
    assigned=np.asarray(assigned,bool)
    if assigned.shape!=p.shape:raise ValueError('Category assignment length differs')
    eligible=assigned & (degrees>0)
    selected=np.flatnonzero(eligible)
    n=len(p)
    detail=pd.DataFrame({'all_neighbor_count':degrees,'strict_majority':p>.5,'assigned_to_group':assigned,'eligible_assigned':eligible,
        'geographically_isolated':degrees==0,'same_group_neighbor_count':np.asarray(W@assigned.astype(int)).ravel(),
        'component_id':np.full(n,-1),'component_size':np.zeros(n,int),'component_eigenvalue':np.full(n,np.nan),
        'component_eigenvector':np.full(n,np.nan),'local_fryer_ssi':np.full(n,np.nan),'eigenpair_residual':np.full(n,np.nan)})
    stats={'assigned_node_count':int(assigned.sum()),'assigned_nodes_without_strict_majority':int((assigned & (p<=.5)).sum()),
        'assigned_degree_zero_excluded':int((assigned & (degrees==0)).sum()),
        'eligible_assigned_node_count':len(selected),'eligible_without_same_group_neighbor_count':int((eligible & (detail.same_group_neighbor_count.to_numpy()==0)).sum())}
    if not len(selected):
        stats.update(fryer_ssi=np.nan,component_count=0,largest_component_size=0,max_eigenpair_residual=0.,status='undefined_no_eligible_assigned_nodes')
        return stats,detail,pd.DataFrame()
    induced=W[selected][:,selected]
    count,labels=connected_components(induced,directed=False)
    components=[];total=0.
    for label in range(count):
        local=np.flatnonzero(labels==label);indices=selected[local]
        component=induced[local][:,local]
        value,v,residual=component_pair(component,degrees[indices])
        total+=value*len(indices)/len(selected)
        detail.loc[indices,'component_id']=label
        detail.loc[indices,'component_size']=len(indices)
        detail.loc[indices,'component_eigenvalue']=value
        detail.loc[indices,'component_eigenvector']=v
        detail.loc[indices,'local_fryer_ssi']=len(indices)*value*v
        detail.loc[indices,'eigenpair_residual']=residual
        components.append({'component_id':label,'component_size':len(indices),'component_eigenvalue':value,'eigenpair_residual':residual})
    np.testing.assert_allclose(detail.loc[eligible,'local_fryer_ssi'].mean(),total,atol=1e-12)
    stats.update(fryer_ssi=total,component_count=count,largest_component_size=max(x['component_size'] for x in components),
        max_eigenpair_residual=max(x['eigenpair_residual'] for x in components),status='ok')
    return stats,detail,pd.DataFrame(components)


def analyze_city(city,scale):
    nodes,metadata=load_nodes(city,scale)
    assignment=dominant_labels(nodes[list(FOUR_CATEGORIES)].to_numpy())
    nodes['dominant_category']=np.array([*FOUR_CATEGORIES,'other_unclassified'])[assignment]
    folder=OUTPUT/city/scale;folder.mkdir(parents=True,exist_ok=True)
    nodes.to_csv(folder/'nodes.csv',index=False)
    summaries=[];local=[];components=[];graph_reports=[]
    for radius in RADII_KM:
        print(f'{CITIES[city]} {scale}: {len(nodes):,} nodes, {radius:g} km',flush=True)
        W=neighbor_graph(nodes.latitude.to_numpy(),nodes.longitude.to_numpy(),radius)
        degree=np.asarray(W.sum(axis=1)).ravel()
        if radius==PRIMARY_RADIUS_KM:
            sparse.save_npz(folder/'adjacency_1km.npz',W)
            R=sparse.diags(np.divide(1,degree,out=np.zeros(len(degree),float),where=degree>0))@W
            sparse.save_npz(folder/'neighbor_weights_1km.npz',R.tocsr())
        graph={'radius_km':radius,'undirected_edges':W.nnz//2,'all_degree_zero_node_count':int((degree==0).sum()),
            'all_degree_zero_population':int(nodes.loc[degree==0,'population'].sum()),'mean_all_neighbor_count':float(degree.mean())}
        graph_reports.append(graph)
        for r,group in enumerate(FOUR_CATEGORIES):
            stats,detail,comp=group_ssi(W,nodes[group].to_numpy()/nodes.population.to_numpy(),assignment==r)
            K=float(nodes[group].sum()/nodes.population.sum())
            result={'city_id':city,'city':CITIES[city],'node_scale':scale,'radius_km':radius,'node_count':len(nodes),
                'city_population':int(nodes.population.sum()),'group':group,'group_population':int(nodes[group].sum()),'population_share_K':K,
                **stats,'fryer_ssi_minus_K':stats['fryer_ssi']-K,'fryer_ssi_normalized':normalize_score(stats['fryer_ssi'],K),
                **{key:value for key,value in graph.items() if key!='radius_km'}}
            summaries.append(result)
            if radius==PRIMARY_RADIUS_KM:
                detail.insert(0,'node_id',nodes.node_id);detail.insert(0,'matrix_index',nodes.matrix_index)
                detail.insert(0,'group',group);detail.insert(0,'node_scale',scale);detail.insert(0,'city_id',city)
                detail['population']=nodes.population;detail['group_population']=nodes[group]
                detail['group_proportion']=nodes[group]/nodes.population
                detail['population_share_K']=K;detail['city_fryer_ssi']=stats['fryer_ssi']
                local.append(detail)
                if len(comp):
                    comp.insert(0,'group',group);components.append(comp)
                idx=np.flatnonzero(detail.eligible_assigned)
                sparse.save_npz(folder/f'B_1km_{group}.npz',R[idx][:,idx])
                np.save(folder/f'B_node_indices_{group}.npy',idx,allow_pickle=False)
    pd.DataFrame(summaries).to_csv(folder/'scores_all_radii.csv',index=False)
    pd.concat(local,ignore_index=True).to_csv(folder/'local_scores_1km.csv.gz',index=False,compression='gzip')
    if components:pd.concat(components,ignore_index=True).to_csv(folder/'components_1km.csv',index=False)
    metadata['graphs']=graph_reports
    (folder/'manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return summaries,metadata


def build():
    rows=[];sources=[]
    for city in CITIES:
        for scale in ['block','tract']:
            scores,metadata=analyze_city(city,scale);rows+=scores;sources.append(metadata)
    results=pd.DataFrame(rows)
    results.to_csv(OUTPUT/'scores_all_radii.csv',index=False)
    primary=results[results.radius_km.eq(PRIMARY_RADIUS_KM)]
    primary[primary.node_scale.eq('block')].to_csv(OUTPUT/'block_scores_1km.csv',index=False)
    primary[primary.node_scale.eq('tract')].to_csv(OUTPUT/'tract_scores_1km.csv',index=False)
    paired=pd.read_csv(BASE/'transit_comparison/paired_comparison.csv')
    keep=['city_id','group','radius_km','node_count','fryer_ssi','fryer_ssi_minus_K','fryer_ssi_normalized','eligible_assigned_node_count','status']
    comparison=paired[[c for c in paired if not c.startswith('fryer_') and c not in ('majority_zone_count',)]].copy()
    for scale in ['block','tract']:
        source=primary[primary.node_scale.eq(scale)][keep].rename(columns={c:'distance_'+scale+'_'+c for c in keep if c not in ('city_id','group')})
        comparison=comparison.merge(source,on=['city_id','group'],validate='one_to_one')
    comparison.to_csv(OUTPUT/'pssi_and_distance_fryer.csv',index=False)
    manifest={'census_year':2020,'geography':'2020 city limits; New York is all five boroughs. No metropolitan suburbs.',
        'primary':'Populated Census blocks, 1 km threshold. Tract-level companion and 0.5/1.5 km sensitivity retain separate labels.',
        'neighbor_rule':'Distinct nodes at great-circle distance <= radius; no self-loops. Earth radius 6378.1 km matching authors neighbors.m. Block model uses Census block internal points; 11-city tract companion uses the existing populated-block representative near each population centroid; NYC tract companion uses Census tract internal points. None are household locations.',
        'weights':'R_ij=1/d_i for each geographic neighbor, where d_i counts ALL populated neighbors. Restrict to assigned-category nodes without renormalization. No population, fare, travel-time, or transit weights.',
        'classification':'Dominant/plurality among White alone, Black alone, Asian alone, Hispanic any race. Largest count wins; ties prefer Hispanic, then Asian, Black, White. This explicit convention matches a 10000-row sample of author published labels, including labels below50%; original preprocessing code unavailable. All-four-zero nodes stay other/unclassified and remain in degree counts. Categories overlap in population counts but each node gets only one label.',
        'isolates':'As authors neighbors.m lines64-69, exclude all-neighbor-degree-zero nodes from the SSI denominator. Nodes with out-group neighbors only remain eligible with local SSI0. No eligible assigned nodes => undefined/NaN, not a fabricated zero.',
        'city_score':'Sum over eligible same-category connected components of (component size / eligible assigned node count) * component Perron eigenvalue.',
        'local_score':'Paper Appendix1: |component| * component eigenvalue * sum-one right eigenvector. Other-category and geographic-isolate local scores undefined. The authors final littlesi data restores the component-size factor omitted from their MATLAB intermediate export.',
        'population_adjustments':'K=all city category residents/all city residents, including nodes excluded from SSI denominator. Requested adjustment=SSI-K. Normalization=(SSI-K)/(1-K). Paper simulated null baseline is not equal to K in general and is not calculated.',
        'limitations':['Census2020 race-alone includes Hispanic; categories overlap.','City boundaries omit neighbors outsidecity; border effects remain.','Equal block weights ignore block population density.','Straight-line neighbor connections may crosswater; no pedestrian routing or transit here.','Tract aggregation and node classification differ from the original 2000 MSA implementation.'],
        'paper_url':PAPER_URL,'author_code_metadata':'data/references/fryer/citations.json','sources':sources,
        'routes_api_calls':0,'network_calls_during_computation':0,'maximum_eigenpair_residual':float(results.max_eigenpair_residual.max())}
    (OUTPUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    write_dictionaries(results,comparison)
    return results,comparison,manifest


def write_dictionaries(results,comparison):
    definitions={
        'city_id':'Stable city directory identifier.', 'city':'2020 city-limit geography; New York includes all five boroughs.',
        'node_scale':'block = populated Census blocks; tract = populated city-limit tract portions. These are separate networks.',
        'radius_km':'Inclusive great-circle distance cutoff in kilometers, with Earth radius6378.1km; distinct nodes only.',
        'node_count':'All positive-population city nodes, before excluding geographic isolates from the SSI denominator.',
        'city_population':'All 2020 city residents; includes those in geographic isolates or other/unclassified nodes.',
        'group':'Category assigned to nodes by the largest of White-alone, Black-alone, Hispanic-any-race, Asian-alone counts. Counts themselves overlap; assigned node labels do not.',
        'group_population':'All city residents in this Census category, irrespective of which category their node was assigned.',
        'population_share_K':'group_population/city_population, fraction0–1. Uses all residents, not the fraction of assigned nodes or only eligible nodes.',
        'assigned_node_count':'Nodes assigned to the category, including geographic isolates.',
        'assigned_nodes_without_strict_majority':'Assigned nodes whose category population share is <=0.5. The dominant-category rule can assign a plurality without a strict majority.',
        'assigned_degree_zero_excluded':'Assigned nodes with no other populated node within the radius. Excluded from the SSI denominator, following author code.',
        'eligible_assigned_node_count':'Assigned nodes with at least one neighbor of any category. Denominator for city SSI component-size weighting.',
        'eligible_without_same_group_neighbor_count':'Eligible assigned nodes with no same-category neighbor. They remain in the SSI denominator and have localSSI0.',
        'fryer_ssi':'Sum over same-category components of component_eigenvalue*component_size/eligible_assigned_node_count. R_ij=1/all-neighbor-degree; restrict R without renormalizing. Undefined if denominator0.',
        'fryer_ssi_minus_K':'Requested population-share adjustment SSI-K. No division. The paper simulated-null adjustment is different and is not computed.',
        'fryer_ssi_normalized':'(SSI-K)/(1-K), using the all-city resident fractionK. Negative values retained; undefined forK1 or undefinedSSI.',
        'component_count':'Number of eligible assigned-category components, including singleton nodes with only other-category neighbors.',
        'largest_component_size':'Largest eligible same-category component node count; zero if no eligible assigned nodes.',
        'max_eigenpair_residual':'Maximum absolute residual |B_C*v_C-lambda_C*v_C| over components. Numerical error, not a segregation score.',
        'status':'ok = at least one eligible assigned node; undefined_no_eligible_assigned_nodes = city SSI and adjustments are blank/NaN.',
        'undirected_edges':'Number of unordered neighboring pairs, including all categories. No self-links.',
        'all_degree_zero_node_count':'All city nodes with no geographic neighbors at this radius, regardless of assigned category.',
        'all_degree_zero_population':'Population of all geographic-isolate nodes. Included in city and group counts used forK.',
        'mean_all_neighbor_count':'Mean degree over all populated nodes, including degree-zero nodes.',
    }
    missing=set(results.columns)-set(definitions)
    if missing:raise ValueError(f'Undocumented score columns: {missing}')
    pd.DataFrame([{'column':c,'definition':definitions[c]} for c in results]).to_csv(OUTPUT/'column_dictionary.csv',index=False)
    local={**definitions,'matrix_index':'Zero-based position in nodes.csv, adjacency and R; B_node_indices maps each category B to these indices.',
        'node_id':'Census GEOID preserved as text:15digits block,11digits tract.',
        'all_neighbor_count':'Number of neighboring nodes of any category within1km, excluding self.',
        'strict_majority':'Whether this category exceeds50% of this node population. Diagnostic only; assignment uses the largest of four counts.',
        'assigned_to_group':'Whether this node is assigned to this category under the dominant-category rule.',
        'eligible_assigned':'Assigned to this category and has at least one geographic neighbor; contributes to the city SSI denominator.',
        'geographically_isolated':'No other populated city node within1km; excluded from SSI, retained in population counts.',
        'same_group_neighbor_count':'Number of1km neighbors assigned to this row category, including for rows not assigned themselves.',
        'component_id':'Zero-based connected-component ID within this city/scale/category. -1 when ineligible.',
        'component_size':'Eligible nodes in this same-category component. Zero when ineligible.',
        'component_eigenvalue':'Perron eigenvalue of induced B for this component; same for every node in the component. Blank when ineligible.',
        'component_eigenvector':'Right Perron vector component, scaled to sum1 within component. Singleton with B=0 uses vector1. Blank when ineligible.',
        'local_fryer_ssi':'component_size*component_eigenvalue*component_eigenvector. Mean over eligible assigned nodes equals citySSI. Can exceed1; no clipping. Blank when ineligible.',
        'eigenpair_residual':'Maximum component absolute residual B*v-lambda*v; repeated for component members.',
        'population':'Total node population, not category population.',
        'group_population':'Number of this category residents in this node, regardless of assigned category.',
        'group_proportion':'Node category population/node total population.',
        'city_fryer_ssi':'City-level category SSI, repeated for interpretation; not a node eigenvalue.'}
    sample=pd.read_csv(OUTPUT/'philadelphia/block/local_scores_1km.csv.gz',nrows=0)
    pd.DataFrame([{'column':c,'definition':local[c]} for c in sample]).to_csv(OUTPUT/'local_column_dictionary.csv',index=False)
    compare={**definitions,'populated_tract_portions':'Node count in the existing travel-cost PSSI model.',
        'population':'All city category residents; same as group_population in the Fryer table.',
        'population_percent':'100*K, for display only; never substituted for fractionK in a normalization.'}
    for metric,formula in [('segregation_index','PSSI dominant city/category eigenvalue lambda'),('segregation_index_normalized','(lambda-K)/(1-K)')]:
        compare[metric+'_with_transit']=formula+' using existing controlled walking+transit model.'
        compare[metric+'_walking_only']=formula+' using existing controlled walking-only model.'
        compare[metric+'_change_without_minus_with']='Walking-only value minus walking+transit value of '+formula+'. Absolute score difference.'
    entries=[]
    for column in comparison:
        if column.startswith('distance_'):
            _,scale,field=column.split('_',2)
            definition=scale.capitalize()+' distance-neighbor model: '+definitions[field]
        else:definition=compare[column]
        entries.append({'column':column,'definition':definition})
    pd.DataFrame(entries).to_csv(OUTPUT/'comparison_column_dictionary.csv',index=False)


def make_plots(results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize,TwoSlopeNorm
    metrics={'fryer_ssi':('Fryer SSI','SSI'),'fryer_ssi_minus_K':('Population-share adjusted Fryer','SSI − K'),'fryer_ssi_normalized':('Normalized Fryer','(SSI − K) / (1 − K)')}
    for scale in ['block','tract']:
        frame=results[results.node_scale.eq(scale)&results.radius_km.eq(PRIMARY_RADIUS_KM)]
        fig,axes=plt.subplots(1,3,figsize=(18,9),layout='constrained')
        for ax,(metric,(title,formula)) in zip(axes,metrics.items()):
            values=frame.pivot(index='city',columns='group',values=metric).reindex(index=CITIES.values(),columns=FOUR_CATEGORIES).to_numpy()
            bound=max(.01,float(np.nanmax(np.abs(values))))
            norm=Normalize(0,1) if metric=='fryer_ssi' else TwoSlopeNorm(vmin=-bound,vcenter=0,vmax=bound)
            cmap=plt.get_cmap('Reds' if metric=='fryer_ssi' else 'RdBu_r').copy();cmap.set_bad('#d4d8dd')
            image=ax.imshow(values,aspect='auto',norm=norm,cmap=cmap)
            ax.set_xticks(range(4),LABELS,fontsize=9);ax.set_yticks(range(len(CITIES)),CITIES.values(),fontsize=10)
            ax.set_title(title+'\n'+formula,fontsize=13,pad=13);ax.tick_params(length=0)
            for i,j in np.ndindex(values.shape):
                rgba=cmap(norm(values[i,j]));brightness=.2126*rgba[0]+.7152*rgba[1]+.0722*rgba[2]
                label=f'{values[i,j]:.3f}' if np.isfinite(values[i,j]) else 'N/A'
                ax.text(j,i,label,ha='center',va='center',fontsize=9,color='white' if brightness<.5 else '#17283b')
            fig.colorbar(image,ax=ax,shrink=.6,pad=.025)
        fig.suptitle(f'Distance-based Fryer SSI · {scale.capitalize()} nodes · 1 km neighbors',fontsize=20)
        fig.supxlabel('2020 city limits; dominant category assigned per node; equal weights across all-race neighbors; no self-links or transit.\nDegree-zero nodes excluded from SSI denominator; K uses all city residents. N/A = no eligible assigned nodes.\nSSI−K is the requested population-share adjustment, not the paper’s simulated null-baseline correction.',fontsize=10)
        fig.savefig(OUTPUT/f'{scale}_fryer_1km.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(15,11),layout='constrained')
    frame=results[results.node_scale.eq('block')]
    palette=plt.get_cmap('tab20').colors
    color_indices=[0,2,4,6,8,10,12,14,16,18,1,3]
    for ax,group,label in zip(axes.ravel(),FOUR_CATEGORIES,LABELS):
        for number,city in enumerate(CITIES):
            rows=frame[frame.city_id.eq(city)&frame.group.eq(group)].sort_values('radius_km')
            ax.plot(rows.radius_km,rows.fryer_ssi,marker='o' if number<10 else '^',
                    linestyle='-' if number<10 else '--',color=palette[color_indices[number]],
                    label=CITIES[city],linewidth=1.4,markersize=4)
        ax.set_title(label.replace('\n',' '));ax.set_xticks(RADII_KM);ax.set_xlabel('Neighbor radius (km)');ax.set_ylabel('Raw Fryer SSI')
        ax.set_ylim(-.02,1.02);ax.grid(alpha=.2)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncols=6,fontsize=10)
    fig.suptitle('Fryer block model: sensitivity to the geographic neighbor radius',fontsize=17)
    fig.savefig(OUTPUT/'radius_sensitivity.png',dpi=180);plt.close(fig)


def write_report(results,comparison,manifest):
    primary=results[results.node_scale.eq('block')&results.radius_km.eq(PRIMARY_RADIUS_KM)]
    columns=['city','group','population_share_K','fryer_ssi','fryer_ssi_minus_K','fryer_ssi_normalized','eligible_assigned_node_count','assigned_degree_zero_excluded','component_count','status']
    table=primary[columns].rename(columns={'population_share_K':'Population fraction K','fryer_ssi':'Fryer SSI','fryer_ssi_minus_K':'SSI − K','fryer_ssi_normalized':'(SSI − K)/(1 − K)','eligible_assigned_node_count':'Eligible assigned blocks','assigned_degree_zero_excluded':'Isolated assigned blocks excluded','component_count':'Components including singletons'})
    html='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Distance-based Fryer and NYC eigenvector maps</title><style>body{font:16px/1.55 system-ui,sans-serif;background:#f4f7fa;color:#182b40;margin:30px}main{max-width:1550px;margin:auto}section{background:white;padding:22px;margin:24px 0;border:1px solid #dbe2ea;border-radius:8px}.note{background:#fff1ce;padding:18px;border-left:4px solid #b98323}img{width:100%;height:auto}.table{overflow:auto;max-height:650px}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:10px;border-bottom:1px solid #dbe2ea;white-space:nowrap;text-align:left}th{background:#eaf0f6;position:sticky;top:0}a{color:#165d91}p{max-width:1200px}</style></head><body><main>
<h1>Distance-based Fryer segregation</h1><p>12 cities including NYC · 2020 Census · four overlapping categories · cached local data · no Routes API calls.</p>
<div class="note">This is a separate distance-neighbor Fryer analysis. The earlier transit-weighted majority-tract calculations remain labeled as a different adaptation. The main results here use Census blocks, with a tract-based companion to expose the effect of node scale.</div>
<p>The <a href="https://eml.berkeley.edu/~fechenique/published/echenique-fryer-qje.pdf">paper’s residential method</a> connects nearby geographic units and assigns equal weight to each neighbor. Here distinct populated nodes within 1 km are neighbors, and Rᵢⱼ=1/dᵢ, where dᵢ counts all neighbors regardless of category. We restrict this matrix to nodes assigned to the category, retaining original weights. There are no self-links, transit weights, or population weights in R.</p>
<p><strong>Assignment rule:</strong> each node receives its largest of the four category counts, even below50%; ties prefer Hispanic, Asian, Black, then White. This follows the pattern in the authors’ released data, whose dominant-category labels differ from a strict &gt;50% reading of “majority” in the paper. The original preprocessing script is unavailable, so our rule is explicit. Nodes with all four counts zero remain other/unclassified. Race-alone populations overlap Hispanic, but each node has one assigned label.</p>
<p>These SSI values describe networks of blocks assigned to a category. Residents living in blocks assigned to another category do not contribute nodes to that category’s SSI, although they remain in K. A zero score indicates no modeled same-category connectivity among eligible assigned nodes; it does not establish that every resident of that race experiences no segregation.</p>
<p>The authors’ code excludes nodes with no geographic neighbors. Eligible nodes with only out-group neighbors have zero local SSI. Each same-category component contributes its Perron eigenvalue weighted by its eligible node count. Its local scores are |component| × eigenvalue × sum-one eigenvector, following the paper. No eligible assigned nodes means undefined, shown as N/A.</p>
<p><strong>Requested columns:</strong> raw SSI; SSI−K; (SSI−K)/(1−K). K uses all city residents, including locations excluded from the SSI denominator. The paper subtracts a simulated baseline; SSI−K is the requested population-share adjustment and is labeled accordingly. This is a 2020 city-limit application with explicitly defined category rules, not a replication of the paper’s 2000 MSA estimates.</p>
<p><a href="block_scores_1km.csv">Main block scores</a> · <a href="tract_scores_1km.csv">Tract companion</a> · <a href="pssi_and_distance_fryer.csv">11-city PSSI and distance-Fryer side-by-side</a> · <a href="scores_all_radii.csv">All threshold results</a> · <a href="column_dictionary.csv">Score column definitions</a> · <a href="comparison_column_dictionary.csv">Comparison column definitions</a> · <a href="local_column_dictionary.csv">Local score definitions</a> · <a href="manifest.json">Methods and source checksums</a></p>
<section><h2>Block-based Fryer: main result</h2><a href="block_fryer_1km.png"><img src="block_fryer_1km.png" alt="Fryer SSI, population-share adjusted SSI and normalized SSI for 12 cities"></a>'''
    html+='<div class="table">'+table.to_html(index=False,border=0,float_format=lambda x:f'{x:.6f}',na_rep='N/A')+'</div></section>'
    html+='''<section><h2>Tract-based companion</h2><img src="tract_fryer_1km.png" alt="Tract-based distance Fryer"><p>Tracts are larger than blocks, so a 1 km rule may leave many without neighbors or without same-category neighbors. These diagnostics are exported; changing aggregation can change the scores substantially. Do not treat the two scales as interchangeable.</p></section>
<section><h2>Radius sensitivity</h2><img src="radius_sensitivity.png" alt="Fryer SSI at 0.5, 1.0 and 1.5 km"><p>Thresholds are fixed equally across cities. Straight-line links can cross water, and city boundaries omit neighbors outside the city. Equal node weighting ignores differences in block population. Categories are evaluated separately because Hispanic overlaps race-alone groups.</p></section>
<section><h2>NYC: four PSSI eigenvector maps</h2><p><a href="../nyc_routes/eigenvector_maps/index.html">Open all four NYC maps and section scores</a>. They use the existing 70-section travel-cost PSSI model; the Fryer block and tract tables above are separate analyses. Colors show70×vᵢ, with section mean1 for each race and a shared logarithmic scale. Raw sum-one eigenvectors are preserved.</p><a href="../nyc_routes/eigenvector_maps/index.html"><img src="../nyc_routes/eigenvector_maps/nyc_four_category_pssi_eigenvectors.png" alt="Four NYC PSSI eigenvector maps"></a></section>
<section><h2>Houston and Phoenix transit diagnostics</h2><p><a href="../multicity/transit_diagnostics/index.html">Coverage and spectral-driver findings</a>. Their small changes are category-specific; substantial changes occur for other categories. A missing modeled boarding connection is not proof that a real transit journey is impossible.</p></section>
<section><h2>Saved matrices and local scores</h2><p>Each city/block or city/tract folder contains ordered nodes, sparse adjacency and neighbor-weight matrices, each category’s induced B matrix and index mapping, component eigenpairs, and local_scores_1km.csv.gz. Local SSI values follow the paper’s component-size scaling. Their average over eligible assigned nodes equals the city SSI.</p><p><a href="verification.json">Independent output verification</a> · <a href="../../data/references/fryer/citations.json">Author-code references and documented interpretation choices</a></p></section></main></body></html>'''
    (OUTPUT/'index.html').write_text(html)


if __name__=='__main__':
    data=build()
    make_plots(data[0])
    write_report(*data)
