"""Dry-run-first, resumable NYC route collection and offline matrix export.

Nothing calls Google unless collect(..., allow_api=True) is explicitly invoked.
Acquisition is per directed pair per mode; there are no automatic retries.
"""
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import math
import os
import sqlite3
import time

import numpy as np
import pandas as pd
from scripts.social_weight import MODES, MissingPriceError, _point, _post, social_weight


@dataclass(frozen=True)
class RouteConfig:
    departure_time: str
    modes: tuple[str, ...] = ('TRANSIT',)

    def payload(self):
        departure = datetime.fromisoformat(self.departure_time.replace('Z', '+00:00'))
        if departure.tzinfo is None or departure.utcoffset() is None:
            raise ValueError('Set an explicit timezone-aware departure time')
        if not self.modes or len(set(self.modes)) != len(self.modes) or not set(self.modes) <= MODES:
            raise ValueError('Provide unique supported travel modes')
        return {'departure_time': departure.astimezone(timezone.utc).isoformat(), 'modes': list(self.modes),
                'request_schema': 1}


def load_zones(path):
    zones = pd.read_csv(path, dtype={'zone_id': str, 'representative_tract': str})
    required = {'matrix_index', 'zone_id', 'latitude', 'longitude', 'TOTPOP'}
    if not required <= set(zones):
        raise ValueError(f'Missing zone columns: {required - set(zones)}')
    if len(zones) < 2 or zones.zone_id.duplicated().any():
        raise ValueError('Need at least two unique zones')
    if zones.matrix_index.tolist() != list(range(len(zones))):
        raise ValueError('matrix_index must be ordered 0,...,n-1')
    if zones[['latitude', 'longitude']].duplicated().any():
        raise ValueError('Representative points must be distinct')
    for coords in zones[['latitude', 'longitude']].itertuples(index=False, name=None):
        _point(coords)
    if not np.isfinite(zones.TOTPOP).all() or (zones.TOTPOP <= 0).any():
        raise ValueError('Zones must have finite positive population')
    return zones


def identity(zones, config):
    # Demographics included to prevent accidentally relabeling a saved run.
    payload = {'zones': json.loads(zones.to_json(orient='records', double_precision=15)),
               'config': config.payload()}
    text = json.dumps(payload, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(text.encode()).hexdigest(), text


def open_cache(path, zones, config):
    fingerprint, payload = identity(zones, config)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    db.execute('CREATE TABLE IF NOT EXISTS metadata (id INTEGER PRIMARY KEY, fingerprint TEXT, payload TEXT)')
    db.execute('CREATE TABLE IF NOT EXISTS responses (i INTEGER, j INTEGER, mode TEXT, state TEXT, response TEXT, updated_at TEXT, PRIMARY KEY(i,j,mode))')
    db.execute('CREATE TABLE IF NOT EXISTS attempts (id INTEGER PRIMARY KEY, i INTEGER, j INTEGER, mode TEXT, started_at TEXT, state TEXT)')
    existing = db.execute('SELECT fingerprint FROM metadata WHERE id=1').fetchone()
    if existing and existing[0] != fingerprint:
        db.close()
        raise ValueError('Cache belongs to different zones, modes or departure time; use a new cache file')
    if not existing:
        db.execute('INSERT INTO metadata VALUES (1,?,?)', (fingerprint, payload))
        db.commit()
    return db


def plan(zones, config):
    config.payload()
    n = len(zones)
    return {'zones': n, 'matrix_shape': [n, n], 'matrix_entries': n*n,
            'local_diagonal_entries': n, 'directed_nonlocal_pairs': n*(n-1),
            'modes': list(config.modes), 'planned_requests': n*(n-1)*len(config.modes),
            'departure_time': config.payload()['departure_time'],
            'api_enabled': False, 'note': 'Count is per departure time, before retries; not a dollar quote.'}


def request_parameters(origin, destination, mode, config):
    body = {'origin': _point(origin), 'destination': _point(destination),
            'travelMode': mode, 'computeAlternativeRoutes': True}
    fields = 'routes.duration,routes.distanceMeters'
    if mode in {'TRANSIT', 'DRIVE'}:
        body['departureTime'] = config.payload()['departure_time'].replace('+00:00', 'Z')
    if mode == 'TRANSIT':
        fields += ',routes.travelAdvisory.transitFare'
    if mode == 'DRIVE':
        body.update(routingPreference='TRAFFIC_AWARE', extraComputations=['TOLLS'])
        fields += ',routes.travelAdvisory.tollInfo'
    if mode in {'WALK', 'BICYCLE'}:
        fields += ',routes.legs.steps.navigationInstruction.maneuver'
    return body, fields


def collect(zones, config, cache_path, *, allow_api=False, max_requests=0,
            api_key=None, interval_seconds=.15, retry_errors=False):
    """Return plan by default. Explicit opt-in AND a positive request cap required.

    Each response is committed immediately. Errors stop the run and remain
    distinguishable from no-route results. Interruptions marked 'inflight' may
    have been billed; they also require retry_errors=True to retry.
    """
    preview = plan(zones, config)
    if not allow_api:
        return preview
    if isinstance(max_requests, bool) or not isinstance(max_requests, int) or max_requests <= 0:
        raise ValueError('Set a positive integer max_requests explicitly')
    if not math.isfinite(interval_seconds) or interval_seconds < 0:
        raise ValueError('interval_seconds must be finite and nonnegative')
    key = api_key or os.environ.get('GOOGLE_MAPS_API_KEY')
    if not key:
        raise ValueError('Set GOOGLE_MAPS_API_KEY locally; never store it in the notebook')
    departure = datetime.fromisoformat(config.payload()['departure_time'])
    now = datetime.now(timezone.utc)
    if 'DRIVE' in config.modes and departure <= now:
        raise ValueError('Driving departure must be in the future')
    if 'TRANSIT' in config.modes and not -7*86400 < (departure-now).total_seconds() < 100*86400:
        raise ValueError('Transit departure must be within the documented past-7/future-100-day window')
    points = zones[['latitude', 'longitude']].to_numpy()
    calls = 0
    db = open_cache(cache_path, zones, config)
    try:
        for i, origin in enumerate(points):
            for j, destination in enumerate(points):
                if i == j:
                    continue
                for mode in config.modes:
                    old = db.execute('SELECT state FROM responses WHERE i=? AND j=? AND mode=?', (i,j,mode)).fetchone()
                    if old and old[0] == 'ok':
                        continue
                    if old and not retry_errors:
                        raise RuntimeError(f'Unresolved request at ({i},{j},{mode}); inspect cache before explicitly retrying')
                    if calls >= max_requests:
                        return {**preview, 'api_enabled': True, 'requests_this_run': calls, 'stopped': 'request_cap'}
                    stamp = datetime.now(timezone.utc).isoformat()
                    cursor = db.execute('INSERT INTO attempts(i,j,mode,started_at,state) VALUES (?,?,?,?,?)', (i,j,mode,stamp,'inflight'))
                    attempt_id = cursor.lastrowid
                    db.execute('INSERT OR REPLACE INTO responses VALUES (?,?,?,?,?,?)', (i,j,mode,'inflight',None,stamp))
                    db.commit()
                    body, fields = request_parameters(origin, destination, mode, config)
                    calls += 1
                    try:
                        response = _post(body, key, fields, 30)
                        if not isinstance(response, dict) or not isinstance(response.get('routes', []), list) or 'error' in response:
                            raise ValueError('Invalid Routes response')
                        serialized = json.dumps(response, allow_nan=False)
                    except Exception:
                        db.execute('UPDATE responses SET state=? WHERE i=? AND j=? AND mode=?', ('error',i,j,mode))
                        db.execute('UPDATE attempts SET state=? WHERE id=?', ('error', attempt_id))
                        db.commit()
                        raise RuntimeError(f'Request failed at ({i},{j},{mode}); progress saved, no automatic retry') from None
                    db.execute('UPDATE responses SET state=?, response=? WHERE i=? AND j=? AND mode=?', ('ok',serialized,i,j,mode))
                    db.execute('UPDATE attempts SET state=? WHERE id=?', ('ok',attempt_id))
                    db.commit()
                    if calls < max_requests:
                        time.sleep(interval_seconds)
        return {**preview, 'api_enabled': True, 'requests_this_run': calls, 'stopped': 'complete'}
    finally:
        db.close()


def export_matrices(zones, config, cache_path, output_dir, *, beta=.1, omega=15., c0=1.25,
                    **price_options):
    """Offline export; pending/error/unknown-price cells stay NaN, never zero.

    duration_seconds and distance_meters refer to the cost-minimizing returned
    route across selected modes. Raw alternatives remain in the SQLite cache.
    A separate fastest-duration tensor is available even when fares are unknown.
    """
    departure = datetime.fromisoformat(config.payload()['departure_time'])
    # Validate economic parameters without networking, including for an empty run.
    social_weight((40,-74), (40,-74), beta=beta, omega=omega, c0=c0,
                  modes=config.modes, departure_time=departure, **price_options)
    if 'DRIVE' in config.modes and price_options.get('driving_usd_per_km') is None:
        raise ValueError('Supply driving_usd_per_km for driving cost matrices')
    n = len(zones)
    arrays = {name: np.full((n,n), np.nan) for name in
              ('duration_seconds', 'distance_meters', 'price_usd', 'cost_usd', 'f', 'log_f')}
    status = np.full((n,n), 'pending', dtype='<U24')
    best_mode = np.full((n,n), '', dtype='<U12')
    fastest = np.full((len(config.modes),n,n), np.nan)
    records = {}
    if Path(cache_path).exists():
        db = open_cache(cache_path, zones, config)
        try:
            records = {(i,j,m):(s,json.loads(r) if r else None)
                       for i,j,m,s,r in db.execute('SELECT i,j,mode,state,response FROM responses')}
        finally:
            db.close()
    points = zones[['latitude','longitude']].to_numpy()
    for i in range(n):
        for j in range(n):
            if i == j:
                for name in ('duration_seconds', 'distance_meters', 'price_usd'):
                    arrays[name][i,j] = 0
                arrays['cost_usd'][i,j], arrays['f'][i,j], arrays['log_f'][i,j] = c0, math.exp(-beta*c0), -beta*c0
                status[i,j], best_mode[i,j] = 'local_model', 'LOCAL'
                fastest[:,i,j] = 0
                continue
            saved = [records.get((i,j,m), ('pending',None)) for m in config.modes]
            for k, (state, response) in enumerate(saved):
                if state == 'ok':
                    try:
                        durations = [float(r['duration'][:-1]) for r in response.get('routes', [])
                                     if isinstance(r.get('duration'), str) and r['duration'].endswith('s')]
                        if len(durations) != len(response.get('routes', [])) or any(not math.isfinite(t) or t < 0 for t in durations):
                            raise ValueError('Invalid duration')
                        fastest[k,i,j] = min(durations, default=math.inf)
                    except (ValueError, TypeError, AttributeError, KeyError):
                        status[i,j] = 'invalid_response'
            if status[i,j] == 'invalid_response':
                continue
            if any(s in {'error', 'inflight'} for s,_ in saved):
                status[i,j] = 'request_error'
                continue
            if any(s == 'pending' for s,_ in saved):
                continue
            responses = {m:r for m,(_,r) in zip(config.modes,saved)}
            try:
                result = social_weight(points[i],points[j], beta=beta, omega=omega,c0=c0,
                    modes=config.modes, departure_time=departure, api_key='offline-replay',
                    request_fn=lambda body,*_: responses[body['travelMode']], **price_options)
            except MissingPriceError:
                status[i,j] = 'price_unknown'
                continue
            except (ValueError, RuntimeError, TypeError, KeyError, AttributeError):
                status[i,j] = 'invalid_response'
                continue
            arrays['cost_usd'][i,j], arrays['f'][i,j], arrays['log_f'][i,j] = result.cost_usd, result.f, result.log_f
            if result.best_mode is None:
                status[i,j] = 'no_route'
                arrays['duration_seconds'][i,j] = arrays['distance_meters'][i,j] = math.inf
            else:
                status[i,j], best_mode[i,j] = 'ok', result.best_mode
                best = min(result.alternatives, key=lambda r:r.generalized_cost_usd)
                arrays['duration_seconds'][i,j], arrays['distance_meters'][i,j], arrays['price_usd'][i,j] = best.hours*3600,best.distance_meters,best.price_usd
    output = Path(output_dir)
    output.mkdir(parents=True,exist_ok=True)
    ids = zones.zone_id.to_numpy(dtype=str)
    np.savez_compressed(output/'matrices.npz', **arrays, status=status, best_mode=best_mode,
                        fastest_duration_seconds_by_mode=fastest, modes=np.array(config.modes),
                        zone_ids=ids, population=zones.TOTPOP.to_numpy())
    for name, array in {**arrays, 'status':status, 'best_mode':best_mode}.items():
        pd.DataFrame(array,index=ids,columns=ids).rename_axis('origin_zone').to_csv(output/f'{name}.csv')
    zones.to_csv(output/'zone_order.csv',index=False)
    unique, counts = np.unique(status,return_counts=True)
    summary = {'status_counts':dict(zip(unique.tolist(),counts.tolist())),
               'configuration':config.payload(), 'beta':beta,'omega_usd_per_hour':omega,'c0_usd':c0,
               'price_options':price_options, 'fingerprint':identity(zones,config)[0],
               'orientation':'row=origin, column=destination; directed, not symmetrized',
               'diagonal':'local cost-floor convention; zero duration is not measured within-zone travel',
               'missing':'NaN=pending/error/unknown price; no_route has f=0, cost=inf',
               'api_requests_made_by_export':0}
    (output/'matrices_manifest.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


def population_transition(log_f, populations):
    """Stable P_ij ∝ exp(log_f_ij)*population_j; reject incomplete matrices."""
    log_f, populations = np.asarray(log_f,float), np.asarray(populations,float)
    if log_f.shape != (len(populations),len(populations)) or np.isnan(log_f).any() or np.isposinf(log_f).any():
        raise ValueError('A complete square log_f matrix is required')
    if not np.isfinite(populations).all() or (populations <= 0).any():
        raise ValueError('Populations must be finite and positive')
    logits = log_f + np.log(populations)[None,:]
    maximum = logits.max(axis=1,keepdims=True)
    shifted = np.full_like(logits,-np.inf)
    np.subtract(logits,maximum,out=shifted,where=np.isfinite(maximum))
    weights = np.exp(shifted)
    return np.divide(weights,weights.sum(axis=1,keepdims=True),out=np.zeros_like(weights),where=weights.sum(axis=1,keepdims=True)>0)
