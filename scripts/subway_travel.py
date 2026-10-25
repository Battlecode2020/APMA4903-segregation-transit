"""Local scheduled subway routing for the 70 groups using the MTA GTFS feed.

Network access is confined to --download. Routing reads the saved ZIP only.
Walking legs are geographic approximations, filtered against land geometry.
"""
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta
from pathlib import Path
from zipfile import ZipFile
from zoneinfo import ZoneInfo
import argparse
import hashlib
import json
import math
import os
import shutil
from urllib.request import urlopen

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer

from scripts.approximate_travel import approximate_matrices, distance_matrix_km
from scripts.nyc_routes import load_zones

ROOT = Path(__file__).resolve().parents[1]
FEED_URL = 'https://rrgtfsfeeds.s3.amazonaws.com/gtfs_subway.zip'
FEED = ROOT/'data/transit/gtfs_subway.zip'


def download_feed(path=FEED):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_suffix('.partial')
    with urlopen(FEED_URL,timeout=90) as source, temporary.open('wb') as target:
        shutil.copyfileobj(source,target)
    with ZipFile(temporary) as archive:
        if archive.testzip() is not None:
            raise ValueError('Corrupt MTA GTFS archive')
    temporary.replace(path)
    path.with_suffix('.download.json').write_text(json.dumps({
        'url':FEED_URL,'downloaded_at':datetime.now(ZoneInfo('America/New_York')).isoformat(),
        'sha256':hashlib.sha256(path.read_bytes()).hexdigest()},indent=2)+'\n')


def seconds(value):
    h,m,s = map(int,str(value).split(':'))
    if h < 0 or not 0 <= m < 60 or not 0 <= s < 60:
        raise ValueError('Invalid GTFS time')
    return h*3600+m*60+s


def active_services(calendar, exceptions, day):
    stamp = day.strftime('%Y%m%d')
    weekday = day.strftime('%A').lower()
    active = set(calendar.loc[(calendar.start_date <= stamp) & (calendar.end_date >= stamp) & (calendar[weekday] == '1'),'service_id'])
    for row in exceptions[exceptions.date == stamp].itertuples():
        if row.exception_type == '1': active.add(row.service_id)
        elif row.exception_type == '2': active.discard(row.service_id)
        else: raise ValueError('Unsupported GTFS calendar exception')
    return active


def load_feed(path, service_date, start_seconds, horizon_seconds):
    with ZipFile(path) as archive:
        if 'frequencies.txt' in archive.namelist():
            raise ValueError('Frequency-based feeds are not supported by this schedule router')
        tables = {name:pd.read_csv(archive.open(name+'.txt'),dtype=str,keep_default_na=False)
                  for name in ('agency','feed_info','calendar','calendar_dates','stops','trips','stop_times','transfers','routes','shapes')}
    if set(tables['agency'].agency_timezone) != {'America/New_York'}:
        raise ValueError('Expected New York agency timezone')
    info = tables['feed_info'].iloc[0].to_dict()
    stamp = service_date.strftime('%Y%m%d')
    if not info['feed_start_date'] <= stamp <= info['feed_end_date']:
        raise ValueError(f'Date {stamp} is outside feed validity {info["feed_start_date"]}–{info["feed_end_date"]}; download a matching feed')
    stops = tables['stops'].copy()
    platforms = stops[stops.parent_station != ''].copy().reset_index(drop=True)
    for column in ('stop_lat','stop_lon'):
        platforms[column] = pd.to_numeric(platforms[column])
    index = dict(zip(platforms.stop_id,range(len(platforms))))
    times = tables['stop_times'].copy()
    for column in ('pickup_type','drop_off_type'):
        if column in times and not times[column].isin(['','0']).all():
            raise ValueError('Feed includes boarding restrictions not supported by this implementation')
    times['arrival'] = times.arrival_time.map(seconds)
    times['departure'] = times.departure_time.map(seconds)
    times['sequence'] = pd.to_numeric(times.stop_sequence)
    if (times.departure < times.arrival).any() or times.duplicated(['trip_id','sequence']).any():
        raise ValueError('Invalid GTFS stop sequence or dwell time')
    all_connections = []
    used_trips = []
    for offset in (-1,0,1):
        day = service_date+timedelta(days=offset)
        services = active_services(tables['calendar'],tables['calendar_dates'],day)
        trips = tables['trips'][tables['trips'].service_id.isin(services)]
        merged = times.merge(trips[['trip_id','route_id','shape_id']],on='trip_id',validate='many_to_one').sort_values(['trip_id','sequence'])
        nxt = merged.groupby('trip_id').shift(-1)
        valid = nxt.stop_id.notna()
        source = merged.loc[valid]
        dest = nxt.loc[valid]
        depart = source.departure.to_numpy()+offset*86400
        arrive = dest.arrival.to_numpy()+offset*86400
        if (arrive < depart).any():
            raise ValueError('Trip moves backward in time')
        mask = (depart >= start_seconds) & (arrive <= start_seconds+horizon_seconds)
        for u,v,dep,arr,trip,route,seq in zip(source.stop_id[mask],dest.stop_id[mask],depart[mask],arrive[mask],source.trip_id[mask],source.route_id[mask],source.sequence[mask]):
            all_connections.append((int(dep),int(arr),index[u],index[v],f'{offset}:{trip}',route,int(seq)))
        used_trips.append(trips)
    # Positive transfer buffers prevent equal-time cross-trip ordering artifacts.
    all_connections.sort(key=lambda c:(c[0],c[4],c[6]))
    transfer_graph = nx.DiGraph()
    transfer_graph.add_nodes_from(range(len(platforms)))
    parent_children = platforms.groupby('parent_station').apply(lambda x:[index[s] for s in x.stop_id],include_groups=False).to_dict()
    self_minimum = np.full(len(platforms),180,dtype=float)
    for row in tables['transfers'].itertuples():
        if row.transfer_type != '2':
            raise ValueError('Expected explicit GTFS minimum transfer times; unsupported transfer type')
        duration = max(120, int(row.min_transfer_time))
        sources = parent_children.get(row.from_stop_id,[index[row.from_stop_id]] if row.from_stop_id in index else [])
        targets = parent_children.get(row.to_stop_id,[index[row.to_stop_id]] if row.to_stop_id in index else [])
        for u in sources:
            for v in targets:
                if u == v: self_minimum[u] = duration
                else:
                    old = transfer_graph.get_edge_data(u,v,{}).get('weight',math.inf)
                    transfer_graph.add_edge(u,v,weight=min(old,duration))
    transfers = {}
    for u in range(len(platforms)):
        distances = nx.single_source_dijkstra_path_length(transfer_graph,u,weight='weight')
        distances[u] = self_minimum[u]
        transfers[u] = list(distances.items())
    return platforms,all_connections,transfers,tables,info,pd.concat(used_trips).drop_duplicates('trip_id')


@dataclass(frozen=True)
class Label:
    time: float
    access: float = 0.
    ride: float = 0.
    wait: float = 0.
    transfer: float = 0.
    legs: tuple = ()


def scan_connections(connections, transfers, access_seconds, start_seconds):
    """Earliest-arrival connection scan; staying on a train avoids reboarding waits."""
    count = len(access_seconds)
    ready = [None]*count
    arrivals = [None]*count
    for u, duration in enumerate(access_seconds):
        if np.isfinite(duration):
            ready[u] = Label(start_seconds+duration,access=duration,legs=(('access',u,float(duration)),))
    onboard = {}
    for dep,arr,u,v,trip,route,sequence in connections:
        label = onboard.get(trip)
        if label is None:
            before = ready[u]
            if before is None or before.time > dep:
                continue
            label = replace(before,time=dep,wait=before.wait+dep-before.time,
                            legs=before.legs+(('board',u,trip,route,float(dep-before.time)),))
        # Includes scheduled intermediate dwell time for passengers staying onboard.
        label = replace(label,time=arr,ride=label.ride+arr-label.time,
                        legs=label.legs+(('ride',u,v,trip,route,dep,arr),))
        onboard[trip] = label
        if arrivals[v] is not None and arrivals[v].time <= arr:
            continue
        arrivals[v] = label
        for target,duration in transfers.get(v,[(v,180)]):
            if ready[target] is None or ready[target].time > arr+duration:
                ready[target] = replace(label,time=arr+duration,transfer=label.transfer+duration,
                    legs=label.legs+(('transfer',v,target,float(duration)),))
    return arrivals


def walking_links(zones, platforms, *, speed_kmh=4.8, detour=1.3, max_access_km=3., max_direct_km=5., entrance_minutes=2.):
    """Approximate walking, rejecting straight segments that visibly cross water.

    The 60 m land buffer tolerates shoreline/generalization differences; this is
    not a street/entrance routing model and may miss valid bridge approaches.
    """
    if not all(math.isfinite(v) for v in (speed_kmh,detour,max_access_km,max_direct_km,entrance_minutes)) or speed_kmh<=0 or detour<1 or min(max_access_km,max_direct_km,entrance_minutes)<0:
        raise ValueError('Invalid walking parameters')
    points = np.vstack([zones[['latitude','longitude']],platforms[['stop_lat','stop_lon']]])
    distances = distance_matrix_km(points)
    n = len(zones)
    geo_access = distances[:n,n:]
    geo_direct = distances[:n,:n]
    transformer = Transformer.from_crs(4326,32618,always_xy=True)
    x,y = transformer.transform(points[:,1],points[:,0])
    xy = np.column_stack([x,y])
    blocks = gpd.read_file(ROOT/'data/raw/tl_2020_36_tabblock20.shp',
        where="COUNTYFP20 IN ('005','047','061','081','085')",columns=['COUNTYFP20','ALAND20','geometry']).to_crs(32618)
    land = blocks.loc[pd.to_numeric(blocks.ALAND20)>0].geometry.union_all().buffer(60)
    shapely.prepare(land)
    access = np.full(geo_access.shape,np.inf)
    i,j = np.nonzero(geo_access*detour <= max_access_km)
    lines = shapely.linestrings(np.stack([xy[i],xy[n+j]],axis=1))
    valid = shapely.covers(land,lines)
    access[i[valid],j[valid]] = 3600*geo_access[i[valid],j[valid]]*detour/speed_kmh+60*entrance_minutes
    direct = np.full((n,n),np.inf)
    i,j = np.nonzero((geo_direct*detour<=max_direct_km) & ~np.eye(n,dtype=bool))
    lines = shapely.linestrings(np.stack([xy[i],xy[j]],axis=1))
    valid = shapely.covers(land,lines)
    direct[i[valid],j[valid]] = 3600*geo_direct[i[valid],j[valid]]*detour/speed_kmh
    np.fill_diagonal(direct,0)
    return access,direct,geo_direct,land


def build(*, feed=FEED, service_date='2026-10-20', departure='08:00:00', horizon_hours=6, beta=.1,omega=15.,c0=1.25):
    if not 0 < horizon_hours <= 24:
        raise ValueError('Horizon must be between 0 and 24 hours')
    if not all(math.isfinite(v) and v>=0 for v in (beta,omega,c0)) or beta==0 or c0==0:
        raise ValueError('Invalid economic parameters')
    zones = load_zones(ROOT/'outputs/nyc_routes/zones.csv')
    if len(zones)!=70: raise ValueError('Expected exactly 70 groups')
    day = date.fromisoformat(service_date)
    start = seconds(departure)
    if start>=86400: raise ValueError('Departure must be a local wall-clock time within the day')
    platforms,connections,transfers,tables,info,used_trips = load_feed(feed,day,start,int(horizon_hours*3600))
    print(f'Loaded {len(platforms)} platform stops and {len(connections)} scheduled connections.',flush=True)
    access,direct,geo,land = walking_links(zones,platforms)
    n = len(zones)
    duration = direct.copy()
    mode = np.where(np.isfinite(direct),'WALK_APPROX','NO_PATH_IN_MODEL').astype('<U24')
    parts = {key:np.full((n,n),np.nan) for key in ('access_seconds','egress_seconds','ride_seconds','wait_seconds','transfer_seconds')}
    paths = {}
    for i in range(n):
        arrivals = scan_connections(connections,transfers,access[i],start)
        for j in range(n):
            if np.isfinite(direct[i,j]):
                for key in parts: parts[key][i,j]=0
                parts['access_seconds'][i,j]=direct[i,j]
            if i == j:
                mode[i,j]='LOCAL'
                continue
            candidates = [(label.time+access[j,p]-start,p,label) for p,label in enumerate(arrivals)
                          if label is not None and np.isfinite(access[j,p]) and label.time+access[j,p]-start<=horizon_hours*3600]
            if not candidates: continue
            elapsed,p,label = min(candidates,key=lambda item:item[0])
            if elapsed < duration[i,j]:
                duration[i,j]=elapsed
                mode[i,j]='SUBWAY_SCHEDULE'
                for key,value in [('access_seconds',label.access),('egress_seconds',access[j,p]),('ride_seconds',label.ride),('wait_seconds',label.wait),('transfer_seconds',label.transfer)]:
                    parts[key][i,j]=value
                paths[i,j] = label.legs+(('egress',p,float(access[j,p])),)
        if (i+1)%10==0: print(f'Computed {i+1}/70 origins.',flush=True)
    covered = np.isfinite(duration)
    if not np.allclose(sum(parts.values())[covered],duration[covered]):
        raise ValueError('Journey time components fail to sum to total')
    output = ROOT/'outputs/nyc_routes/subway_network'/f'{service_date}_{departure.replace(":", "")}'
    output.mkdir(parents=True,exist_ok=True)
    ids = zones.zone_id.to_numpy(dtype=str)
    baseline = approximate_matrices(zones[['latitude','longitude']].to_numpy(),beta=beta,omega=omega,c0=c0)
    hybrid_duration = np.where(covered,duration,baseline['duration_seconds'])
    hybrid_mode = mode.copy()
    hybrid_mode[~covered]='DISTANCE_FALLBACK'
    for name,times,which_mode in [('subway_only',duration,mode),('hybrid',hybrid_duration,hybrid_mode)]:
        folder = output/name
        folder.mkdir(exist_ok=True)
        costs = np.maximum(c0,omega*times/3600)
        costs[~np.isfinite(times)]=np.nan  # no path in this limited model is not proof of no real route
        log_f = -beta*costs
        status = np.where(np.isfinite(times),'approximate','no_path_in_model').astype('<U24')
        np.fill_diagonal(status,'local_model')
        arrays = {'duration_minutes':times/60,'duration_seconds':times,'cost_usd':costs,'log_f':log_f,'f':np.exp(log_f),
                  'status':status,'best_mode':which_mode,'distance_meters':np.full((n,n),np.nan),
                  'great_circle_distance_km':geo,'network_covered':covered,**parts}
        # Distance-only fallbacks have no subway itinerary or component breakdown.
        np.savez_compressed(folder/'matrices.npz',**arrays,zone_ids=ids,population=zones.TOTPOP.to_numpy(),source=np.array(name))
        for field in ('duration_minutes','best_mode','status','network_covered','cost_usd','f'):
            pd.DataFrame(arrays[field],index=ids,columns=ids).rename_axis('origin_zone').to_csv(folder/f'{field}.csv')
    records=[]
    for i in range(n):
        for j in range(n):
            raw = paths.get((i,j),())
            legs=[]
            for leg in raw:
                if leg[0]=='ride':
                    _,u,v,trip,route,dep,arr=leg
                    if legs and legs[-1]['type']=='ride' and legs[-1]['trip_id']==trip:
                        legs[-1].update(to_stop=platforms.iloc[v].stop_id,arrival_seconds=arr)
                    else:
                        legs.append({'type':'ride','from_stop':platforms.iloc[u].stop_id,'to_stop':platforms.iloc[v].stop_id,'trip_id':trip,'route_id':route,'departure_seconds':dep,'arrival_seconds':arr})
                elif leg[0]=='board':
                    _,u,trip,route,wait=leg
                    legs.append({'type':'wait','stop_id':platforms.iloc[u].stop_id,'seconds':wait})
                elif leg[0]=='transfer':
                    _,u,v,secs=leg
                    legs.append({'type':'transfer','from_stop':platforms.iloc[u].stop_id,'to_stop':platforms.iloc[v].stop_id,'seconds':secs})
                else:
                    typ,u,secs=leg
                    legs.append({'type':typ,'stop_id':platforms.iloc[u].stop_id,'seconds':secs})
            records.append({'origin':ids[i],'destination':ids[j],'mode':hybrid_mode[i,j],
                            'minutes':float(hybrid_duration[i,j]/60),'network_covered':bool(covered[i,j]),'legs':legs})
    with (output/'journeys.jsonl').open('w') as stream:
        for row in records: stream.write(json.dumps(row)+'\n')
    stations = tables['stops'][tables['stops'].location_type=='1'].copy()
    stations.to_csv(output/'stations.csv',index=False)
    tables['routes'].to_csv(output/'routes.csv',index=False)
    zones.to_csv(output/'zone_order.csv',index=False)
    service_shapes = set(used_trips.shape_id)
    shapes = tables['shapes'][tables['shapes'].shape_id.isin(service_shapes)].copy()
    for column in ('shape_pt_lat','shape_pt_lon','shape_pt_sequence'):
        shapes[column]=pd.to_numeric(shapes[column])
    route_shapes=used_trips.groupby('shape_id').route_id.agg(lambda s:','.join(sorted(set(s)))).to_dict()
    features=[]
    for sid,points in shapes.groupby('shape_id'):
        points=points.sort_values('shape_pt_sequence')
        features.append({'type':'Feature','properties':{'shape_id':sid,'routes':route_shapes[sid]},
            'geometry':{'type':'LineString','coordinates':points[['shape_pt_lon','shape_pt_lat']].values.tolist()}})
    (output/'subway_lines.geojson').write_text(json.dumps({'type':'FeatureCollection','features':features}))
    coverage = pd.DataFrame({'zone_id':ids,'borough':zones.borough,
        'candidate_platforms':np.isfinite(access).sum(axis=1),
        'network_destinations_excluding_self':covered.sum(axis=1)-1,
        'distance_fallback_destinations':(~covered).sum(axis=1)})
    coverage.to_csv(output/'coverage_by_group.csv',index=False)
    metadata={'groups':70,'matrix_shape':[70,70],'service_date':service_date,'departure_local':departure,'timezone':'America/New_York',
        'horizon_hours':horizon_hours,'feed_info':info,'feed_url':FEED_URL,'feed_sha256':hashlib.sha256(Path(feed).read_bytes()).hexdigest(),
        'calendar_applied':True,'scheduled_connections':len(connections),'network_covered_nonlocal_pairs':int(covered.sum()-70),
        'distance_fallback_pairs':int((~covered).sum()),'subway_pairs':int((mode=='SUBWAY_SCHEDULE').sum()),
        'direct_walk_pairs':int((mode=='WALK_APPROX').sum()),'routes_api_calls':0,'routing_network_requests':0,
        'walking':{'speed_kmh':4.8,'detour_factor':1.3,'maximum_access_or_egress_km':3,'maximum_direct_walk_km':5,'entrance_or_exit_minutes':2,'land_buffer_meters':60},
        'transfers':'GTFS minimum transfer times, floored at 120 seconds; same-platform reboarding default 180 seconds; staying onboard incurs no transfer penalty.',
        'cost_model':{'beta':beta,'omega_usd_per_hour':omega,'c0_usd':c0,'fare':'not modeled; time-only generalized cost'},
        'limitations':['2020 demographics combined with the stated 2026 scheduled service, not a reconstruction of 2020 transit.',
            'Regular MTA schedule feed; most temporary changes, realtime delays and disruptions are not represented.',
            'Subway and Staten Island Railway only; no buses, ferries or commuter rail.',
            'Walking uses distance and a land mask, not streets or verified station entrances; real bridges may be rejected.',
            'No path in the capped subway/walking model does not mean no real-world route exists.',
            'Hybrid fills uncovered pairs with the previous uncalibrated distance model; inspect best_mode and coverage.',
            'Full path distance is not computed; distance_meters stays NaN. Group-level times depend on one representative point.']}
    (output/'manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
    make_map(zones,platforms,features,land,output)
    return output,metadata


def make_map(zones,platforms,features,land,output):
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'data/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    boundaries=gpd.read_file(ROOT/'outputs/nyc_routes/zones.geojson').to_crs(32618)
    boundaries.geometry=boundaries.geometry.intersection(land)
    lines=gpd.GeoDataFrame.from_features(features,crs=4326).to_crs(32618)
    fig,ax=plt.subplots(figsize=(11,12))
    boundaries.plot(ax=ax,color='#f1f5f9',edgecolor='#94a3b8',linewidth=.45)
    lines.plot(ax=ax,color='#197568',linewidth=.7,alpha=.65)
    parents=platforms.drop_duplicates('parent_station')
    gpd.GeoSeries(gpd.points_from_xy(parents.stop_lon,parents.stop_lat),crs=4326).to_crs(32618).plot(ax=ax,color='#197568',markersize=3)
    xy=gpd.GeoSeries(gpd.points_from_xy(zones.longitude,zones.latitude),crs=4326).to_crs(32618)
    xy.plot(ax=ax,color='#c2410c',markersize=12)
    for zone,point in zip(zones.zone_id,xy): ax.annotate(zone,(point.x,point.y),fontsize=6,xytext=(3,2),textcoords='offset points')
    ax.set_title('70 analysis groups and actual MTA subway/SIR line shapes\nOrange: group points • green: GTFS lines and stations',fontsize=13)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(output/'subway_groups_map.png',dpi=180)
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download',action='store_true',help='Explicitly download the public MTA feed first')
    parser.add_argument('--date',default='2026-10-20')
    parser.add_argument('--departure',default='08:00:00')
    args=parser.parse_args()
    if args.download: download_feed()
    output,report=build(service_date=args.date,departure=args.departure)
    print(json.dumps({'output':str(output),'coverage':report['network_covered_nonlocal_pairs'],'fallback':report['distance_fallback_pairs']},indent=2))
