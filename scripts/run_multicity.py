"""Compute all city travel matrices and scores from already-downloaded data."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback

import geopandas as gpd
import shapely

from scripts.multicity_analysis import BASE, ROOT, CITIES, build_all
from scripts.multicity_transit import TransitParameters, compute_city


def run(city_ids=None,force=False):
    catalog=json.loads((BASE/'transit_sources.json').read_text())
    sources={city['city_id']:city for city in catalog['cities']}
    completed=[];failures={};start=time.perf_counter()
    for city in city_ids or CITIES:
        print(f'Routing {CITIES[city]}...',flush=True)
        city_start=time.perf_counter()
        try:
            entry=sources[city]
            paths=[]
            for feed in entry['feeds']:
                path=ROOT/feed['path']
                if hashlib.sha256(path.read_bytes()).hexdigest()!=feed['sha256']:
                    raise ValueError(f'Feed checksum differs: {path}')
                paths.append(path)
            folder=BASE/city
            water=gpd.read_file(folder/'demographics/water.geojson').geometry.union_all()
            shapely.prepare(water)
            params=TransitParameters(reference_date=entry.get('recommended_service_date') or '2026-10-20')
            report=compute_city(folder/'demographics/nodes.csv',paths,folder/'travel',parameters=params,water_geometry=water,force=force)
            completed.append(city)
            print(f'{CITIES[city]}: {report["transit_reachable_nonlocal_pairs"]:,}/{report["nonlocal_pairs"]:,} transit pairs; {time.perf_counter()-city_start:.1f}s',flush=True)
        except Exception as error:
            failures[city]={'error':str(error),'traceback':traceback.format_exc()}
            print(f'{CITIES[city]} needs repair: {error}',flush=True)
    (BASE/'run_status.json').write_text(json.dumps({'completed':completed,'failures':failures,'elapsed_seconds':time.perf_counter()-start,'routes_api_calls':0},indent=2)+'\n')
    if completed:build_all(completed)
    if failures:raise RuntimeError(f'Cities needing repair: {list(failures)}; see run_status.json')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cities',nargs='+',choices=CITIES)
    parser.add_argument('--force',action='store_true')
    args=parser.parse_args()
    run(args.cities,args.force)
