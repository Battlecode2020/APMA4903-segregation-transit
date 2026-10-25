"""Offline distance-based travel-time scenarios for the existing 70 NYC groups.

This is an uncalibrated spatial approximation, not road or transit routing.
No network access, API credentials, or additional downloads are used.
"""
from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json
import math

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Approximation:
    effective_speed_kmh: float = 18.0
    detour_factor: float = 1.4
    access_wait_minutes: float = 10.0

    def validate(self):
        values = asdict(self)
        if not all(math.isfinite(v) for v in values.values()):
            raise ValueError('Approximation parameters must be finite')
        if self.effective_speed_kmh <= 0 or self.detour_factor < 1 or self.access_wait_minutes < 0:
            raise ValueError('Speed must be positive, detour factor at least 1, and overhead nonnegative')


# Assumed sensitivity scenarios, not empirical estimates or confidence bounds.
SCENARIOS = {
    'faster': Approximation(25, 1.2, 5),
    'baseline': Approximation(18, 1.4, 10),
    'slower': Approximation(12, 1.6, 15),
}


def distance_matrix_km(coordinates):
    """Great-circle distances, input rows are (latitude, longitude) degrees."""
    points = np.asarray(coordinates, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not len(points):
        raise ValueError('Provide an n by 2 array of latitude/longitude coordinates')
    if not np.isfinite(points).all() or (abs(points[:,0]) > 90).any() or (abs(points[:,1]) > 180).any():
        raise ValueError('Invalid WGS84 coordinates')
    lat, lon = np.radians(points).T
    a = np.sin((lat[:,None]-lat[None,:])/2)**2
    a += np.cos(lat[:,None])*np.cos(lat[None,:])*np.sin((lon[:,None]-lon[None,:])/2)**2
    result = 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(a,0,1)))
    np.fill_diagonal(result,0)
    return result


def approximate_matrices(coordinates, parameters=Approximation(), *, beta=.1, omega=15., c0=1.25, trip_cost_usd=0.):
    parameters.validate()
    if not all(math.isfinite(v) and v >= 0 for v in (beta,omega,c0,trip_cost_usd)) or beta == 0 or c0 == 0:
        raise ValueError('Require finite nonnegative costs and omega, and positive beta/c0')
    direct = distance_matrix_km(coordinates)
    adjusted = parameters.detour_factor * direct
    minutes = 60 * adjusted / parameters.effective_speed_kmh + parameters.access_wait_minutes
    np.fill_diagonal(minutes,0)  # local encounters are handled by c0, not measured travel
    price = np.full_like(minutes,trip_cost_usd)
    np.fill_diagonal(price,0)
    cost = np.maximum(c0,price + omega*minutes/60)
    log_f = -beta*cost
    status = np.full(minutes.shape,'approximate',dtype='<U24')
    np.fill_diagonal(status,'local_model')
    return {'great_circle_distance_km':direct, 'assumed_path_distance_km':adjusted,
            'duration_minutes':minutes,'duration_seconds':minutes*60,
            'distance_meters':adjusted*1000,'price_usd':price,'cost_usd':cost,
            'f':np.exp(log_f),'log_f':log_f,'status':status}


def export(zones, output_dir, *, scenarios=None, beta=.1, omega=15., c0=1.25, trip_cost_usd=0.):
    """Export scenario matrices in exactly the input zones' order."""
    scenarios = SCENARIOS if scenarios is None else scenarios
    if not scenarios:
        raise ValueError('Provide at least one scenario')
    if zones.zone_id.duplicated().any() or zones.matrix_index.tolist() != list(range(len(zones))):
        raise ValueError('Zone IDs must be unique and matrix_index must be ordered')
    for name in scenarios:
        if not name or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_-' for c in name):
            raise ValueError('Scenario names must be simple lowercase file names')
    output = Path(output_dir)
    points = zones[['latitude','longitude']].to_numpy()
    report = {'method':'great_circle_distance_times_detour_divided_by_speed_plus_overhead',
        'formula':'minutes_ij = 60 * detour_factor * great_circle_km_ij / speed_kmh + access_wait_minutes (i != j)',
        'group_count':len(zones),'matrix_shape':[len(zones),len(zones)],
        'matrix_entries':len(zones)**2,'routes_api_calls':0,'network_requests':0,
        'zone_ids':zones.zone_id.tolist(),
        'zone_definition_sha256':hashlib.sha256(zones.to_json(orient='records',double_precision=15).encode()).hexdigest(),
        'economics':{'beta':beta,'omega_usd_per_hour':omega,'c0_usd':c0,'trip_cost_usd':trip_cost_usd},
        'cost_interpretation':'Time-only cost by default (trip_cost_usd=0), not a free-transit claim; set a justified price separately.',
        'orientation':'row=origin, column=destination; approximation is symmetric',
        'diagonal':'Zero travel time by convention; generalized cost is c0. Within-group travel is not estimated.',
        'limitations':['Uncalibrated assumptions; scenarios are sensitivity tests, not confidence bounds.',
            'No street network, bridge/tunnel constraints, ferry schedules, subway lines, traffic, transfers, or departure-time effects.',
            'Especially weak for water crossings and Staten Island; straight-line proximity does not ensure accessibility.',
            'One representative point per group; within-group variation is not represented.',
            'All distinct groups are treated as reachable, unlike a network routing model.'],
        'scenarios':{}}
    for name, parameters in scenarios.items():
        arrays = approximate_matrices(points,parameters,beta=beta,omega=omega,c0=c0,trip_cost_usd=trip_cost_usd)
        folder = output/name
        folder.mkdir(parents=True,exist_ok=True)
        ids = zones.zone_id.to_numpy(dtype=str)
        np.savez_compressed(folder/'matrices.npz',**arrays,zone_ids=ids,population=zones.TOTPOP.to_numpy(),
                            source=np.array('offline_approximation'))
        for field in ('duration_minutes','great_circle_distance_km','assumed_path_distance_km','cost_usd','f','log_f','status'):
            pd.DataFrame(arrays[field],index=ids,columns=ids).rename_axis('origin_zone').to_csv(folder/f'{field}.csv')
        off_diagonal = arrays['duration_minutes'][~np.eye(len(zones),dtype=bool)]
        report['scenarios'][name] = {**asdict(parameters),
            'minimum_nonlocal_minutes':float(off_diagonal.min()),
            'median_nonlocal_minutes':float(np.median(off_diagonal)),
            'maximum_nonlocal_minutes':float(off_diagonal.max())}
    zones.to_csv(output/'zone_order.csv',index=False)
    (output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def build():
    from scripts.nyc_routes import load_zones
    base = ROOT/'outputs/nyc_routes'
    zones = load_zones(base/'zones.csv')
    if len(zones) != 70:
        raise ValueError('This NYC analysis requires exactly 70 groups')
    race = pd.read_csv(base/'demographics_2020/race_counts_2020.csv')
    if race.zone_id.tolist() != zones.zone_id.tolist():
        raise ValueError('Race counts and routing groups have different orders')
    groups = zones[['matrix_index','zone_id','borough','tract_count','latitude','longitude']].merge(
        race.drop(columns=['matrix_index','borough']),on='zone_id',validate='one_to_one',sort=False)
    if not np.array_equal(groups.total_population,zones.TOTPOP):
        raise ValueError('Population mismatch')
    groups.to_csv(base/'groups_70.csv',index=False)
    return export(zones,base/'offline_approximation')


if __name__ == '__main__':
    print(json.dumps(build(),indent=2))
