#!/usr/bin/env python3
"""Build New York 2020 Census dual graphs from TIGER and PL 94-171 data."""
from pathlib import Path
import argparse, gc, hashlib, json, re, time, warnings
from zipfile import ZipFile
import geopandas as gpd
import pandas as pd
import numpy as np
import networkx as nx
import shapely
import maup
from gerrychain import Graph
from importlib.metadata import version

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/ny_2020'
RAW = ROOT / 'data/raw'
SUP = ROOT / 'data/supplements'
CRS = 'EPSG:5070'
LEVELS = {'counties':'county','county_subdivisions':'cousub','tracts':'tract','block_groups':'bg','blocks':'tabblock'}
RACES = ['WHITE','BLACK','AMIN','ASIAN','NHPI','OTHER','2MORE']
VAPS = ['WVAP','BVAP','AMINVAP','ASIANVAP','NHPIVAP','OTHERVAP','2MOREVAP']
POP = ['TOTPOP'] + ['NH_'+x for x in RACES] + ['HISP'] + ['H_'+x for x in RACES] + ['VAP','HVAP'] + VAPS
ALLPOP = POP + ['U'+x for x in POP] + ['R'+x for x in POP]

def log(s): print(time.strftime('%H:%M:%S'), s, flush=True)
def save_json(path, obj): path.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()

def demographics():
    """Use official Census layouts; join segments by LOGRECNO, never row order."""
    with ZipFile(SUP/'2020PL_R_import_scripts.zip') as z:
        script=z.read('R Versions/pl_geohd_2020_dar.r').decode()
    names=re.findall(r'"([A-Z0-9]+)"',script.split('colnames(header) <- c(')[1])
    needed=['LOGRECNO','SUMLEV','GEOCOMP','GEOCODE','STATE','COUNTY','COUSUB','TRACT','BLKGRP','BLOCK','POP100']
    with ZipFile(SUP/'ny2020.pl.zip') as z:
        with z.open('nygeo2020.pl') as f:
            geo=pd.read_csv(f,sep='|',header=None,names=names,usecols=needed,dtype=str,keep_default_na=False)
        assert geo.LOGRECNO.is_unique
        geo=geo[geo.GEOCOMP.eq('00') & geo.SUMLEV.isin(['040','050','060','140','150','750'])].copy()
        geo=geo.set_index('LOGRECNO')
        segments=[]
        for seg, tables in [(1,[1,2]),(2,[3,4])]:
            cols=['FILEID','STUSAB','CHARITER','CIFSN','LOGRECNO']
            for t in tables: cols += [f'P{t:03d}{i:04d}' for i in range(1,72 if t%2 else 74)]
            if seg==2: cols += ['H0010001','H0010002','H0010003']
            use=['LOGRECNO']+[f'P{t:03d}{i:04d}' for t in tables for i in range(1,12 if t%2==0 else 10)]
            with z.open(f'ny0000{seg}2020.pl') as f:
                data=pd.read_csv(f,sep='|',names=cols,header=None,usecols=use,dtype={'LOGRECNO':str})
            assert data.LOGRECNO.is_unique
            segments.append(data.set_index('LOGRECNO').reindex(geo.index))
    d=pd.concat(segments,axis=1)
    assert d.notna().all().all()
    pop=pd.DataFrame(index=d.index)
    pop['TOTPOP']=d.P0010001
    assert (pop.TOTPOP==d.P0020001).all()
    assert (pop.TOTPOP==geo.POP100.astype(int)).all()
    pop['HISP']=d.P0020002
    for i,r in enumerate(RACES):
        pop['NH_'+r]=d[f'P002{i+5:04d}']
        pop['H_'+r]=d[f'P001{i+3:04d}']-pop['NH_'+r]
    pop['VAP']=d.P0030001
    assert (pop.VAP==d.P0040001).all()
    pop['HVAP']=d.P0040002
    for i,r in enumerate(VAPS): pop[r]=d[f'P004{i+5:04d}']
    assert (pop>=0).all().all()
    assert (pop[['NH_'+r for r in RACES]].sum(axis=1)+pop.HISP==pop.TOTPOP).all()
    assert (pop[['H_'+r for r in RACES]].sum(axis=1)==pop.HISP).all()
    assert (pop[VAPS].sum(axis=1)+pop.HVAP==pop.VAP).all()
    return geo.join(pop[POP])

def read_layer(kind):
    x=gpd.read_file(RAW/f'tl_2020_36_{kind}20.shp')
    assert x.GEOID20.is_unique and x.GEOID20.notna().all()
    assert not x.geometry.is_empty.any() and x.geometry.notna().all()
    invalid=~x.geometry.is_valid
    if invalid.any(): raise ValueError(f'{kind}: {invalid.sum()} invalid geometries; explicit review required')
    return x

def build(level, blocks, block_pop, official):
    kind=LEVELS[level]; log(f'{level}: reading and assigning blocks')
    x=blocks if level=='blocks' else read_layer(kind)
    out=OUT/level; out.mkdir(parents=True,exist_ok=True)
    qa={'level':level,'nodes':len(x),'invalid_geometries':0}
    if level=='blocks':
        values=block_pop[ALLPOP].copy(); assignment=pd.Series(blocks.index,index=blocks.index)
    else:
        # Native NAD83 coordinates preserve the exact nesting in the supplied TIGER layers.
        assignment=maup.assign(blocks.geometry,x.geometry).reindex(blocks.index)
        assert assignment.notna().all(), f'{level}: unassigned blocks'
        assignment=assignment.astype(int)
        values=block_pop[ALLPOP].groupby(assignment).sum().reindex(x.index,fill_value=0)
        assigned_geo=assignment.map(x.GEOID20)
        if level=='county_subdivisions': expected=block_pop['COUSUB_GEOID20']
        else: expected=blocks.GEOID20.str[:{'counties':5,'tracts':11,'block_groups':12}[level]]
        bad=assigned_geo.ne(expected)
        qa['assignment_disagreements_with_census']=int(bad.sum())
        if bad.any():
            pd.DataFrame({'GEOID20':blocks.GEOID20[bad],'spatial':assigned_geo[bad],'census':expected[bad]}).to_csv(out/'assignment_disagreements.csv',index=False)
            raise ValueError(f'{level}: spatial assignment differs from Census; review required')
        cross=pd.DataFrame({'BLOCKID20':blocks.GEOID20,'id':assignment,'GEOID20':assigned_geo})
        cross.to_csv(out/'block_assignment.csv.gz',index=False)
    assert (values.sum()==block_pop[ALLPOP].sum()).all()
    for c in POP: assert (values[c]==values['U'+c]+values['R'+c]).all()
    summary={'counties':'050','county_subdivisions':'060','tracts':'140','block_groups':'150','blocks':'750'}[level]
    off=official[official.SUMLEV.eq(summary)].set_index('GEOCODE')[POP]
    assert off.index.is_unique
    expected=off.reindex(x.GEOID20)
    assert expected.notna().all().all()
    assert np.array_equal(values[POP].to_numpy(),expected.to_numpy()), f'{level}: published population mismatch'
    qa['population_checks']='All 25 demographic totals match published Census units; all U + R identities hold'
    qa['TOTPOP']=int(values.TOTPOP.sum());qa['VAP']=int(values.VAP.sum())
    qa['UTOTPOP']=int(values.UTOTPOP.sum());qa['RTOTPOP']=int(values.RTOTPOP.sum())
    projected=x.to_crs(CRS)
    centers=projected.geometry.centroid
    # Keep source attributes, but replace blank urban labels with the matched official supplement.
    attrs=x.drop(columns='geometry').copy()
    attrs['area']=attrs.ALAND20.astype('int64')
    attrs['geometry_area']=projected.geometry.area
    attrs['C_X']=centers.x; attrs['C_Y']=centers.y
    if level=='blocks':
        attrs['BLOCKID20']=attrs.GEOID20
        attrs['UR20']=block_pop.UR20
        attrs['UACE20']=block_pop.UACE20.fillna('')
        attrs['UATYPE20']=block_pop.UATYPE20.fillna('')
    attrs=pd.concat([attrs,values],axis=1).replace({np.nan:None})
    log(f'{level}: GerryChain rook adjacency ({len(x):,} polygons)')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        graph=Graph.from_geodataframe(projected[['geometry']],adjacency='rook',cols_to_add=[])
    graph_nx=graph.get_nx_graph()
    warn=[str(w.message) for w in caught]
    save_json(out/'warnings.json',warn)
    if any('overlap' in w.lower() for w in warn):
        raise ValueError(f'{level}: overlapping projected polygons; review warnings.json')
    for i,row in attrs.iterrows():
        graph_nx.nodes[i].update(row.to_dict())
        graph_nx.nodes[i].setdefault('boundary_perim',0.0)
    graph_nx.graph.update({'state':'New York','state_fips':'36','census_year':2020,'level':level,
        'measurement_crs':CRS,'length_units':'metres','area_units':'square metres',
        'area_definition':'ALAND20 from supplied Census shapefile',
        'centroid_definition':'geometric centroid in EPSG:5070',
        'adjacency':'rook; positive length shared boundary; no artificial island connections',
        'population_source':'2020 Census PL 94-171 P1, P2, P3, P4',
        'urban_source':'Census TIGER2020 tabblock20 updated 2023; joined by GEOID20'})
    assert graph_nx.number_of_nodes()==len(x)
    assert nx.number_of_selfloops(graph_nx)==0
    assert all(np.isfinite(d['shared_perim']) and d['shared_perim']>0 for _,_,d in graph_nx.edges(data=True))
    assert all(d['boundary_perim']>=-1e-5 for _,d in graph_nx.nodes(data=True))
    components=sorted(nx.connected_components(graph_nx),key=lambda c:(-len(c),min(c)))
    component_id={n:k for k,c in enumerate(components) for n in c}
    qa.update(edges=graph_nx.number_of_edges(),components=len(components),
        isolates=list(nx.isolates(graph_nx)),component_sizes=[len(c) for c in components],
        boundary_nodes=sum(d['boundary_node'] for _,d in graph_nx.nodes(data=True)))
    log(f'{level}: saving {qa["edges"]:,} edges')
    graph.to_json(str(out/'graph.json'))
    nodes=pd.DataFrame.from_dict(dict(graph_nx.nodes(data=True)),orient='index').drop(columns='geometry',errors='ignore')
    nodes.index.name='id'; nodes['component']=pd.Series(component_id);nodes['degree']=pd.Series(dict(graph_nx.degree()))
    nodes.to_csv(out/'nodes.csv',index=True)
    edges=pd.DataFrame([(min(a,b),max(a,b),d['shared_perim']) for a,b,d in graph_nx.edges(data=True)],columns=['source','target','shared_perim']).sort_values(['source','target'])
    edges['source_geoid']=edges.source.map(x.GEOID20);edges['target_geoid']=edges.target.map(x.GEOID20)
    edges.to_csv(out/'edges.csv',index=False)
    save_json(out/'validation.json',qa)
    # A light preview retains all units; the graph calculations use unsimplified shapes.
    preview=x[['GEOID20','geometry']].copy().to_crs(CRS)
    preview.geometry=preview.geometry.simplify(250,preserve_topology=True)
    if level=='counties':
        preview['NAME20']=x.NAME20;preview['TOTPOP']=values.TOTPOP
        preview.to_crs(4326).to_file(OUT/'counties_preview.geojson',driver='GeoJSON')
    del graph,graph_nx,projected,attrs,nodes,edges,preview
    gc.collect()
    return qa

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--levels',nargs='+',choices=LEVELS,default=list(LEVELS));a=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    log('Loading block population from Census PL 94-171')
    official=demographics(); blocks=read_layer('tabblock')
    off=official[official.SUMLEV.eq('750')].set_index('GEOCODE')
    assert set(off.index)==set(blocks.GEOID20)
    b=off.reindex(blocks.GEOID20).reset_index(drop=True)
    b['COUSUB_GEOID20']=b.STATE+b.COUNTY+b.COUSUB
    urban=gpd.read_file(SUP/'urban/tl_2020_36_tabblock20.dbf',ignore_geometry=True).set_index('GEOID20')
    assert urban.index.is_unique and set(urban.index)==set(blocks.GEOID20)
    u=urban.reindex(blocks.GEOID20).reset_index(drop=True)
    assert (u.POP20.to_numpy()==b.TOTPOP.to_numpy()).all()
    for col in ['UR20','UACE20','UATYPE20']:b[col]=u[col]
    assert b.UR20.isin(['U','R']).all()
    for prefix in ['U','R']:
        b=pd.concat([b,b[POP].mul(b.UR20.eq(prefix),axis=0).add_prefix(prefix)],axis=1)
    log(f'All {len(blocks):,} blocks matched; population {b.TOTPOP.sum():,}; urban labels complete')
    report={}
    for level in a.levels: report[level]=build(level,blocks,b,official)
    for level in LEVELS:
        p=OUT/level/'validation.json'
        if p.exists():report[level]=json.loads(p.read_text())
    save_json(OUT/'validation.json',report)
    manifest={'created_utc':pd.Timestamp.now(tz='UTC').isoformat(),'census_year':2020,
      'software':{p:version(p) for p in ['gerrychain','maup','geopandas','shapely','networkx','pandas','pyproj']},
      'inputs':{str(p.relative_to(ROOT)):sha(p) for p in [SUP/'ny2020.pl.zip',SUP/'tl_2020_36_tabblock20_updated.zip',SUP/'2020PL_R_import_scripts.zip']},
      'geometry_files':{str(p.relative_to(ROOT)):sha(p) for p in sorted(RAW.glob('*')) if p.suffix in ['.shp','.dbf','.prj','.shx']}}
    save_json(OUT/'manifest.json',manifest)
    log('Build complete')
if __name__=='__main__':main()
