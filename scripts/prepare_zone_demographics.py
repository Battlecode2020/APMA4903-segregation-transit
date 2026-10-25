"""Verify and export 2020 Census P1/P2 counts for the fixed 70 NYC zones.

Reads only local Census legacy files. No networking or Routes API calls.
"""
from pathlib import Path
from zipfile import ZipFile
import hashlib
import json
import re

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COUNTIES = ['005', '047', '061', '081', '085']
RACES = ['WHITE', 'BLACK', 'AMIN', 'ASIAN', 'NHPI', 'OTHER', '2MORE']
NAMES = ['white_alone', 'black_alone', 'american_indian_alaska_native_alone',
         'asian_alone', 'native_hawaiian_other_pacific_islander_alone',
         'some_other_race_alone', 'two_or_more_races']
LABELS = ['White alone', 'Black or African American alone',
          'American Indian and Alaska Native alone', 'Asian alone',
          'Native Hawaiian and Other Pacific Islander alone',
          'Some Other Race alone', 'Two or more races']
P1 = [f'P001{i:04d}' for i in range(3, 10)]
P2 = [f'P002{i:04d}' for i in range(5, 12)]
COLUMNS = ['P0010001', *P1, 'P0020001', 'P0020002', *P2]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_official():
    """Join geographic headers to P1/P2 by LOGRECNO, not file row order."""
    supplements = ROOT / 'data/supplements'
    with ZipFile(supplements / '2020PL_R_import_scripts.zip') as archive:
        layout = archive.read('R Versions/pl_geohd_2020_dar.r').decode()
    names = re.findall(r'"([A-Z0-9]+)"', layout.split('colnames(header) <- c(')[1])
    with ZipFile(supplements / 'ny2020.pl.zip') as archive:
        with archive.open('nygeo2020.pl') as stream:
            geo = pd.read_csv(stream, sep='|', header=None, names=names,
                              usecols=['LOGRECNO','SUMLEV','GEOCOMP','GEOCODE','STATE','COUNTY','POP100'],
                              dtype=str, keep_default_na=False)
        geo = geo[(geo.STATE == '36') & geo.COUNTY.isin(COUNTIES) &
                  (geo.GEOCOMP == '00') & geo.SUMLEV.isin(['050','140'])]
        names = ['FILEID','STUSAB','CHARITER','CIFSN','LOGRECNO']
        names += [f'P001{i:04d}' for i in range(1,72)]
        names += [f'P002{i:04d}' for i in range(1,74)]
        with archive.open('ny000012020.pl') as stream:
            data = pd.read_csv(stream, sep='|', header=None, names=names,
                               usecols=['LOGRECNO',*COLUMNS], dtype={'LOGRECNO':str})
    official = geo.merge(data, on='LOGRECNO', how='left', validate='one_to_one')
    require(official[COLUMNS].notna().all().all(), 'Missing official Census record')
    require((official[COLUMNS] >= 0).all().all(), 'Negative Census count')
    require(np.array_equal(official.P0010001, official.P0020001), 'P1/P2 population mismatch')
    require(np.array_equal(official.P0010001, official.POP100.astype('int64')), 'Geographic population mismatch')
    require(np.array_equal(official[P1].sum(axis=1), official.P0010001), 'P1 race categories do not sum to population')
    require(np.array_equal(official[P2].sum(axis=1) + official.P0020002, official.P0010001), 'P2 categories do not sum to population')
    return official


def build():
    base = ROOT / 'outputs/nyc_routes'
    output = base / 'demographics_2020'
    zones = pd.read_csv(base / 'zones.csv')
    cross = pd.read_csv(base / 'tract_to_zone.csv', dtype=str)
    official = read_official()
    tracts = official[official.SUMLEV == '140'].set_index('GEOCODE')
    counties = official[official.SUMLEV == '050'].set_index('GEOCODE')
    require(len(zones) == 70 and zones.zone_id.is_unique, 'Expected 70 unique zones')
    require(zones.matrix_index.tolist() == list(range(70)), 'Zone ordering differs from route matrices')
    require(cross.GEOID20.is_unique, 'Tract assigned more than once')
    require(set(cross.GEOID20) == set(tracts.index), 'Crosswalk must include every NYC tract exactly once')
    require(set(cross.zone_id) == set(zones.zone_id), 'Crosswalk zone IDs do not match')
    joined = cross.merge(tracts[COLUMNS], left_on='GEOID20', right_index=True, validate='one_to_one')
    grouped = joined.groupby('zone_id')[COLUMNS].sum().reindex(zones.zone_id)
    require(grouped.notna().all().all(), 'Zone missing demographic values')
    counts = grouped[P1].to_numpy(dtype=np.int64)
    nh = grouped[P2].to_numpy(dtype=np.int64)
    hispanic_by_race = counts - nh
    hispanic = grouped.P0020002.to_numpy(dtype=np.int64)
    population = grouped.P0010001.to_numpy(dtype=np.int64)
    require((hispanic_by_race >= 0).all(), 'Negative derived Hispanic race count')
    require(np.array_equal(hispanic_by_race.sum(axis=1), hispanic), 'Hispanic race counts do not sum correctly')
    require(np.array_equal(zones.TOTPOP, population), 'Zone population mismatch')
    # Independently check every existing tract value and zone value against raw P1/P2.
    local = pd.read_csv(ROOT / 'outputs/ny_2020/tracts/nodes.csv', dtype={'GEOID20':str}).set_index('GEOID20').loc[tracts.index]
    require(np.array_equal(local.TOTPOP, tracts.P0010001), 'Existing tract populations differ from official data')
    require(np.array_equal(local.HISP, tracts.P0020002), 'Existing tract Hispanic counts differ')
    for k, race in enumerate(RACES):
        require(np.array_equal(local[f'NH_{race}'], tracts[P2[k]]), f'Existing tract NH_{race} differs')
        require(np.array_equal(local[f'H_{race}'] + local[f'NH_{race}'], tracts[P1[k]]), f'Existing tract {race} differs')
        require(np.array_equal(zones[f'NH_{race}'], nh[:,k]), f'Zone NH_{race} differs')
        require(np.array_equal(zones[f'H_{race}'], hispanic_by_race[:,k]), f'Zone H_{race} differs')
    require(np.array_equal(zones.HISP, hispanic), 'Zone Hispanic count mismatch')
    by_county = tracts.assign(county=tracts.index.str[:5]).groupby('county')[COLUMNS].sum()
    require(np.array_equal(by_county, counties.loc[by_county.index,COLUMNS]), 'Tract aggregates differ from published county totals')
    require(int(population.sum()) == 8804190, 'Unexpected NYC 2020 total population')

    output.mkdir(parents=True, exist_ok=True)
    meta = zones[['matrix_index','zone_id','borough']].copy()
    meta['total_population'] = population
    race_frame = pd.concat([meta, pd.DataFrame(counts, columns=NAMES)], axis=1)
    race_frame.to_csv(output / 'race_counts_2020.csv', index=False)
    ethnicity_names = ['hispanic_latino_any_race', *['non_hispanic_' + n for n in NAMES]]
    ethnicity = np.column_stack([hispanic, nh])
    ethnicity_frame = pd.concat([meta,pd.DataFrame(ethnicity, columns=ethnicity_names)],axis=1)
    ethnicity_frame.to_csv(output / 'race_ethnicity_counts_2020.csv', index=False)
    rows = []
    for i, zone in enumerate(zones.zone_id):
        for origin, values in [('not_hispanic_latino',nh),('hispanic_latino',hispanic_by_race)]:
            for k, name in enumerate(NAMES):
                rows.append({'matrix_index':i,'zone_id':zone,'hispanic_origin':origin,
                             'race':name,'count':int(values[i,k]),'fraction_of_zone':float(values[i,k]/population[i])})
    pd.DataFrame(rows).to_csv(output / 'race_by_hispanic_origin_2020.csv',index=False)
    np.savez_compressed(output / 'demographics_2020.npz',
        zone_ids=zones.zone_id.to_numpy(dtype=str), population=population,
        race_names=np.array(NAMES), race_counts=counts, race_proportions=counts/population[:,None],
        race_ethnicity_names=np.array(ethnicity_names), race_ethnicity_counts=ethnicity,
        race_ethnicity_proportions=ethnicity/population[:,None],
        hispanic_origin_names=np.array(['not_hispanic_latino','hispanic_latino']),
        race_by_hispanic_origin_counts=np.stack([nh,hispanic_by_race],axis=1))
    city = pd.DataFrame({'race':NAMES,'label':LABELS,'count':counts.sum(axis=0),
                         'percent_of_nyc':100*counts.sum(axis=0)/population.sum()})
    city.to_csv(output / 'nyc_race_totals_2020.csv',index=False)
    race_frame.groupby('borough')[['total_population',*NAMES]].sum().to_csv(output/'borough_race_totals_2020.csv')
    dictionary = {'population_universe':'All persons, all ages, April 1, 2020',
        'race_categories':{name:{'label':label,'P1_variable':p1,'non_hispanic_P2_variable':p2}
                           for name,label,p1,p2 in zip(NAMES,LABELS,P1,P2)},
        'total_population_variable':'P0010001 (equals P0020001)',
        'hispanic_latino_any_race_variable':'P0020002',
        'definitions':['P1 race counts include both Hispanic and non-Hispanic persons.',
            'Six race-alone categories plus two-or-more-races form seven disjoint groups summing to total population.',
            'A race-alone category excludes people reporting that race together with another race.',
            'The separate race/ethnicity table contains Hispanic persons of any race plus seven non-Hispanic race groups; its eight groups are disjoint.',
            'Hispanic counts for each race equal P1 race count minus corresponding P2 non-Hispanic count.',
            'Counts are published Census P.L. 94-171 values subject to Census disclosure avoidance; no interpolation or prisoner reallocation applied.'],
        'array_axes':{'race_counts':'zone, race (70 x 7)',
                      'race_ethnicity_counts':'zone, race/ethnicity (70 x 8)',
                      'race_by_hispanic_origin_counts':'zone, Hispanic origin, race (70 x 2 x 7)'},
        'matrix_alignment':'zone_ids and matrix_index match zones.csv and all route matrix rows/columns'}
    (output/'data_dictionary.json').write_text(json.dumps(dictionary,indent=2)+'\n')
    report = {'census_year':2020,'zone_count':70,'tract_count':len(tracts),'nyc_population':int(population.sum()),
        'all_checks_passed':True,
        'checks':['Every NYC tract assigned exactly once',
                  'All existing tract population and race/Hispanic counts match raw P1/P2',
                  'All existing zone population and race/Hispanic counts match direct raw-file aggregation',
                  'All race and race/ethnicity categories sum to zone population',
                  'All tract aggregates match separately published county records',
                  'Zone order matches route matrix order'],
        'sources':{'legacy_zip':'https://www2.census.gov/programs-surveys/decennial/2020/data/01-Redistricting_File--PL_94-171/New_York/ny2020.pl.zip',
                   'P1':'https://data.census.gov/table/DECENNIALPL2020.P1',
                   'P2':'https://data.census.gov/table/DECENNIALPL2020.P2'},
        'sha256':{str(p.relative_to(ROOT)):sha256(p) for p in [ROOT/'data/supplements/ny2020.pl.zip',ROOT/'data/supplements/2020PL_R_import_scripts.zip',base/'tract_to_zone.csv',base/'zones.csv']},
        'routes_api_calls':0,'network_requests':0,'used_2010_data':False}
    (output/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__ == '__main__':
    print(json.dumps(build(),indent=2))
