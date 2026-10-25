"""Build 70 borough-preserving NYC zones entirely from local 2020 Census data."""
from pathlib import Path
import hashlib
import json
import os

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[1]
BOROUGHS = {'005': ('BX', 'Bronx'), '047': ('BK', 'Brooklyn'),
            '061': ('MN', 'Manhattan'), '081': ('QN', 'Queens'),
            '085': ('SI', 'Staten Island')}


def partition(frame, count):
    """Population-balanced recursive spatial bisection of whole tracts.

    Split along the population-weighted principal axis. Deterministic tie
    breaking uses GEOID order. This favors compactness, not guaranteed rook
    contiguity; no artificial road or bridge links are inferred.
    """
    if count == 1:
        return [frame]
    xy = frame[['x', 'y']].to_numpy()
    weights = frame.TOTPOP.to_numpy()
    center = np.average(xy, axis=0, weights=weights)
    delta = xy - center
    _, vectors = np.linalg.eigh((delta * weights[:, None]).T @ delta)
    axis = vectors[:, -1]
    if axis[np.argmax(np.abs(axis))] < 0:
        axis = -axis
    order = np.argsort(delta @ axis, kind='stable')
    ordered = frame.iloc[order]
    left_count = count // 2
    right_count = count - left_count
    sums = ordered.TOTPOP.cumsum().to_numpy()
    # Reserve at least one populated tract for each eventual leaf.
    positive = (ordered.TOTPOP.to_numpy() > 0).cumsum()
    valid = np.flatnonzero((positive >= left_count) &
                          (positive[-1] - positive >= right_count))
    cut = valid[np.argmin(abs(sums[valid] - sums[-1] * left_count / count))] + 1
    return partition(ordered.iloc[:cut], left_count) + partition(ordered.iloc[cut:], right_count)


def build(output=ROOT / 'outputs/nyc_routes', n=70):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    source = ROOT / 'outputs/ny_2020/tracts/nodes.csv'
    nodes = pd.read_csv(source, dtype={'GEOID20': str, 'COUNTYFP20': str})
    nodes = nodes[nodes.COUNTYFP20.isin(BOROUGHS)].sort_values('GEOID20').copy()
    shapes = gpd.read_file(ROOT / 'data/raw/tl_2020_36_tract20.shp')
    shapes = shapes[['GEOID20', 'geometry']].merge(nodes, on='GEOID20', validate='one_to_one')
    shapes = shapes.to_crs(32618)  # local metric coordinates for partitioning
    transform = Transformer.from_crs(4326, 32618, always_xy=True)
    shapes['x'], shapes['y'] = transform.transform(shapes.INTPTLON20, shapes.INTPTLAT20)
    populations = shapes.groupby('COUNTYFP20').TOTPOP.sum().sort_index()
    quota = n * populations / populations.sum()
    allocations = np.floor(quota).astype(int)
    for county in (quota - allocations).sort_values(ascending=False, kind='stable').index[:n - allocations.sum()]:
        allocations[county] += 1
    if (allocations < 1).any():
        raise ValueError('Choose enough zones to represent all five boroughs')
    demo = list(nodes.columns[nodes.columns.get_loc('TOTPOP'):nodes.columns.get_loc('boundary_perim')])
    records, memberships = [], []
    for county, count in allocations.items():
        prefix, borough = BOROUGHS[county]
        pieces = partition(shapes[shapes.COUNTYFP20 == county], int(count))
        pieces.sort(key=lambda f: (f.y.mean(), f.x.mean()))
        for number, piece in enumerate(pieces, 1):
            zone_id = f'{prefix}{number:02d}'
            center = np.average(piece[['x', 'y']], axis=0, weights=piece.TOTPOP)
            populated = piece[piece.TOTPOP > 0]
            # Snap to a populated tract's internal point, not a centroid in water.
            selected = populated.iloc[np.argmin(((populated[['x', 'y']] - center)**2).sum(axis=1))]
            geom = piece.geometry.union_all()
            record = {'zone_id': zone_id, 'borough': borough, 'tract_count': len(piece),
                      'latitude': float(selected.INTPTLAT20), 'longitude': float(selected.INTPTLON20),
                      'representative_tract': selected.GEOID20,
                      'polygon_parts': len(geom.geoms) if hasattr(geom, 'geoms') else 1,
                      **{c: int(piece[c].sum()) for c in demo}, 'geometry': geom}
            records.append(record)
            memberships.extend({'GEOID20': g, 'zone_id': zone_id} for g in piece.GEOID20)
    zones = gpd.GeoDataFrame(records, crs=shapes.crs)
    zones.insert(0, 'matrix_index', np.arange(n))
    assert len(zones) == n and (zones.TOTPOP > 0).all()
    assert len(memberships) == len(nodes)
    assert len({m['GEOID20'] for m in memberships}) == len(nodes)
    for col in demo:
        assert zones[col].sum() == nodes[col].sum(), col
    assert not zones[['latitude', 'longitude']].duplicated().any()
    zones.drop(columns='geometry').to_csv(output / 'zones.csv', index=False)
    pd.DataFrame(memberships).sort_values('GEOID20').to_csv(output / 'tract_to_zone.csv', index=False)
    zones.to_crs(4326).to_file(output / 'zones.geojson', driver='GeoJSON')
    summary = {'zone_count': n, 'tract_count': len(nodes), 'population_2020': int(nodes.TOTPOP.sum()),
               'borough_allocations': {BOROUGHS[k][1]: int(v) for k, v in allocations.items()},
               'zone_population_min': int(zones.TOTPOP.min()), 'zone_population_max': int(zones.TOTPOP.max()),
               'zone_population_mean': float(zones.TOTPOP.mean()),
               'source_nodes_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
               'method': 'Whole-tract population-balanced recursive spatial bisection within each borough',
               'representative': 'Populated tract internal point nearest population-weighted mean of tract internal points',
               'limitations': ['Custom analytical zones, not official neighborhoods.',
                              'Contiguity and street access are not guaranteed; review map and routing points.',
                              'One point approximates a zone; within-zone travel uses the local cost model.',
                              'Demographics describe 2020, not current population.'],
               'api_requests_made': 0}
    (output / 'zones_manifest.json').write_text(json.dumps(summary, indent=2) + '\n')
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'data/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    fig, ax = plt.subplots(figsize=(12, 12))
    cmap = ListedColormap(plt.get_cmap('tab20').colors)
    # Mask the display to land-containing Census blocks; analytical tract
    # boundaries stay intact, including their water areas.
    blocks = gpd.read_file(ROOT / 'data/raw/tl_2020_36_tabblock20.shp',
                          where="COUNTYFP20 IN ('005','047','061','081','085')",
                          columns=['COUNTYFP20', 'ALAND20', 'geometry']).to_crs(shapes.crs)
    blocks = blocks[pd.to_numeric(blocks.ALAND20) > 0]
    if blocks.empty:
        raise ValueError('No NYC land blocks available for the map')
    display = zones.copy()
    display.geometry = zones.geometry.intersection(blocks.geometry.union_all())
    display['color'] = (np.arange(n) * 7) % 20
    display.plot(ax=ax, column='color', cmap=cmap, edgecolor='#454545', linewidth=.5, alpha=.8)
    for row in zones.itertuples():
        x, y = transform.transform(row.longitude, row.latitude)
        ax.plot(x, y, '.', color='#111827', markersize=3)
        ax.annotate(row.zone_id, (x, y), fontsize=6, xytext=(2, 2), textcoords='offset points')
    ax.set_title('NYC: 70 population-balanced zones\nWhole 2020 Census tracts; dots mark proposed routing points', fontsize=16)
    ax.set_axis_off()
    fig.text(.5, .025, 'Custom zones • display masked to land-containing blocks • no Routes API calls', ha='center')
    fig.tight_layout(rect=[0, .04, 1, 1])
    fig.savefig(output / 'zones_map.png', dpi=180)
    plt.close(fig)
    return summary


if __name__ == '__main__':
    print(json.dumps(build(), indent=2))
