"""Download, cache, inspect and inventory agency-produced static GTFS schedules.

No routing service is called. Network access requires the explicit --download flag.
WMATA/Phoenix use identified public MobilityData archive copies; producer URLs and
mirror provenance are both retained. Feed dates are data, never silently extended.
"""
from __future__ import annotations
import argparse
import collections
import csv
import datetime as dt
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/multicity/transit'
MANIFEST = ROOT / 'outputs/multicity/transit_sources.json'
TARGET_DATE = dt.date(2026, 10, 20)
REQUIRED = {'agency.txt', 'routes.txt', 'stops.txt', 'trips.txt', 'stop_times.txt'}
REQUIRED_COLUMNS = {
    'agency.txt': {'agency_name', 'agency_url', 'agency_timezone'},
    'routes.txt': {'route_id', 'route_type'},
    'stops.txt': {'stop_id', 'stop_lat', 'stop_lon'},
    'trips.txt': {'route_id', 'service_id', 'trip_id'},
    'stop_times.txt': {'trip_id', 'arrival_time', 'departure_time', 'stop_id', 'stop_sequence'},
}

def feed(feed_id, agency, source_url, download_url, **kwargs):
    return dict(feed_id=feed_id, agency=agency, source_url=source_url,
                download_url=download_url, path=f'data/multicity/transit/{feed_id}.zip',
                notes=[], **kwargs)

SOURCES = [
 dict(city_id='philadelphia', city_name='Philadelphia', feeds=[
  feed('septa_bus','SEPTA Bus, Metro and Trolley','https://www3.septa.org/developer/','https://www3.septa.org/developer/gtfs_public.zip',archive_member='google_bus.zip'),
  feed('septa_rail','SEPTA Regional Rail','https://www3.septa.org/developer/','https://www3.septa.org/developer/gtfs_public.zip',archive_member='google_rail.zip')], missing_agencies=['PATCO'], coverage_notes=['SEPTA local bus, subway, trolley and regional rail. PATCO and NJ Transit are not included.']),
 dict(city_id='miami',city_name='Miami',feeds=[
  feed('miami_dade','Miami-Dade Transit','https://www.miamidade.gov/global/transportation/open-data-feeds.page','https://www.miamidade.gov/transit/googletransit/current/google_transit.zip'),
  feed('tri_rail','South Florida Regional Transportation Authority / Tri-Rail','https://gtfs.tri-rail.com/','https://gtfs.tri-rail.com/gtfs.zip')],missing_agencies=['Miami municipal trolley','Brightline'],coverage_notes=['Miami-Dade Metrobus, Metrorail and Metromover plus Tri-Rail. Metrorail is coded GTFS rail type 2 by the producer. Municipal trolleys and intercity Brightline omitted.']),
 dict(city_id='los_angeles',city_name='Los Angeles',feeds=[
  feed('la_metro_bus','LA Metro bus','https://lacmta.github.io/gtfs-bus/','https://gitlab.com/LACMTA/gtfs_bus/raw/master/gtfs_bus.zip'),
  feed('la_metro_rail','LA Metro rail','https://developer.metro.net/gtfs-schedule-data/','https://gitlab.com/LACMTA/gtfs_rail/raw/master/gtfs_rail.zip'),
  feed('metrolink','Metrolink','https://metrolinktrains.com/about/','https://metrolinktrains.com/globalassets/about/gtfs/gtfs.zip')],missing_agencies=['LADOT DASH and Commuter Express','Municipal bus operators'],coverage_notes=['Metro local bus, BRT, subway and light rail, plus Metrolink commuter rail. LADOT and other municipal bus agencies omitted.']),
 dict(city_id='salt_lake_city',city_name='Salt Lake City',feeds=[
  feed('uta','Utah Transit Authority','https://gtfsfeed.rideuta.com/','https://gtfsfeed.rideuta.com/GTFS.zip')],missing_agencies=[],coverage_notes=['UTA bus, TRAX light rail, S-Line streetcar and FrontRunner commuter rail. Route and departure data provided by permission of UTA.']),
 dict(city_id='san_francisco',city_name='San Francisco',feeds=[
  feed('sfmta','SFMTA Muni','https://www.sfmta.com/reports/gtfs-transit-data','https://muni-gtfs.apps.sfmta.com/data/muni_gtfs-current.zip'),
  feed('bart','Bay Area Rapid Transit','https://www.bart.gov/schedules/developers/gtfs','https://www.bart.gov/dev/schedules/google_transit.zip'),
  feed('caltrain','Caltrain','https://www.caltrain.com/developer-resources','https://data.trilliumtransit.com/gtfs/caltrain-ca-us/caltrain-ca-us.zip')],missing_agencies=['Golden Gate Transit','SamTrans','AC Transit'],coverage_notes=['Muni buses, light rail, historic streetcars and cable cars; BART; Caltrain. Regional bus agencies and ferries omitted. The Muni official current feed ends 2026-08-28, so this city uses an explicitly historical valid weekday.']),
 dict(city_id='washington_dc',city_name='Washington DC',feeds=[
  feed('wmata_bus','WMATA Metrobus','https://www.wmata.com/about/developers.html','https://files.mobilitydatabase.org/mdb-1846/mdb-1846-202609100152/mdb-1846-202609100152.zip',producer_url='https://api.wmata.com/gtfs/bus-gtfs-static.zip',mirror_source_url='https://mobilitydatabase.org/feeds/gtfs/mdb-1846'),
  feed('wmata_rail','WMATA Metrorail','https://www.wmata.com/about/developers.html','https://files.mobilitydatabase.org/mdb-1847/mdb-1847-202610160108/mdb-1847-202610160108.zip',producer_url='https://api.wmata.com/gtfs/rail-gtfs-static.zip',mirror_source_url='https://mobilitydatabase.org/feeds/gtfs/mdb-1847')],missing_agencies=['DC Streetcar','MARC','VRE','Other regional bus operators'],coverage_notes=['WMATA bus and rail agency-produced files obtained from public MobilityData archives, without API keys. DC Streetcar available public GTFS expired 2025-06-30; omitted rather than extending its dates. MARC/VRE and other regional buses omitted.']),
 dict(city_id='houston',city_name='Houston',feeds=[
  feed('houston_metro','Houston METRO','https://www.ridemetro.org/about/business-to-business/developer-portal','https://ridemetro-gtfs.azurewebsites.us/google_transit.zip')],missing_agencies=[],coverage_notes=['METRO local and express bus plus METRORail light rail.']),
 dict(city_id='detroit',city_name='Detroit',feeds=[
  feed('ddot','Detroit Department of Transportation','https://data.detroitmi.gov/','https://www.detroitmi.gov/Portals/0/docs/deptoftransportation/pdfs/ddot_gtfs.zip'),
  feed('qline','QLINE','https://www.qlinedetroit.com/','https://s3.amazonaws.com/etatransit.gtfs/qline.etaspot.net/gtfs.zip'),
  feed('dpm_current','Detroit People Mover','https://www.thepeoplemover.com/','https://hosted-gtfs-feeds.s3.amazonaws.com/DPM/gtfs.zip'),
  feed('smart','SMART regional bus','https://apps1.smartbus.org/gtfs/','https://apps1.smartbus.org/gtfs/smart_gtfs.zip')],missing_agencies=['D2A2 and DAX regional express'],coverage_notes=['DDOT and SMART buses, QLINE streetcar and Detroit People Mover. QLINE is classified as route_type 2 by its producer.']),
 dict(city_id='chicago',city_name='Chicago',feeds=[
  feed('cta','Chicago Transit Authority','https://www.transitchicago.com/developers/gtfs/','https://www.transitchicago.com/downloads/sch_data/google_transit.zip'),
  feed('metra','Metra','https://metra.com/gtfs-realtime-api-key-request-license-agreement','https://schedules.metrarail.com/gtfs/schedule.zip')],missing_agencies=['Pace','South Shore Line'],coverage_notes=['CTA buses and L trains plus Metra commuter rail. Pace and South Shore omitted.']),
 dict(city_id='phoenix',city_name='Phoenix',feeds=[
  feed('phoenix','Valley Metro','https://www.phoenixopendata.com/dataset/valley-metro-bus-schedule','https://files.mobilitydatabase.org/mdb-147/mdb-147-202609290147/mdb-147-202609290147.zip',producer_url='https://phoenixopendata.com/dataset/3eae9a4a-98b9-40c8-8df7-8c00c1756235/resource/28ccc0a5-49c8-495c-b91f-193de5ce2cb7/download/googletransit.zip',mirror_source_url='https://mobilitydatabase.org/feeds/gtfs/mdb-147')],missing_agencies=[],coverage_notes=['Valley Metro bus, light rail and Tempe streetcar. Agency-produced file from public MobilityData archive because city download returned HTTP 403. Data attribution: City of Phoenix / Valley Metro (ODC-By).']),
 dict(city_id='atlanta',city_name='Atlanta',feeds=[
  feed('marta','MARTA','https://itsmarta.com/app-developer-resources.aspx','https://itsmarta.com/google_transit_feed/google_transit.zip')],missing_agencies=['CobbLinc','Ride Gwinnett','Xpress'],coverage_notes=['MARTA bus, heavy rail and Atlanta Streetcar. Suburban agencies omitted.'])
]


def table_members(z):
    """Resolve nested-directory tables while refusing ambiguous duplicate names."""
    result = {}
    for name in z.namelist():
        base = Path(name).name.lower()
        if not base.endswith('.txt') or name.startswith('__MACOSX/'):
            continue
        if base in result:
            raise ValueError(f'Duplicate GTFS table {base}')
        result[base] = name
    return result


def rows(z, members, name):
    if name not in members:
        return []
    with z.open(members[name]) as binary:
        with io.TextIOWrapper(binary, encoding='utf-8-sig') as stream:
            return [{str(k).strip(): v.strip() if isinstance(v, str) else v
                     for k, v in row.items()} for row in csv.DictReader(stream)]


def parse_date(value):
    return dt.datetime.strptime(value, '%Y%m%d').date()


def active_services(cal, exceptions, date):
    day = date.strftime('%Y%m%d')
    weekday = ['monday','tuesday','wednesday','thursday','friday','saturday','sunday'][date.weekday()]
    active = {r['service_id'] for r in cal if r['start_date'] <= day <= r['end_date'] and r.get(weekday) == '1'}
    for r in exceptions:
        if r['date'] == day:
            if r['exception_type'] == '1': active.add(r['service_id'])
            elif r['exception_type'] == '2': active.discard(r['service_id'])
    return active


def inspect_feed(path, target=TARGET_DATE):
    """Structural inspection, calendar evidence and checksum; not formal GTFS validation."""
    with zipfile.ZipFile(path) as z:
        members = table_members(z)
        missing = REQUIRED - members.keys()
        if missing or not {'calendar.txt','calendar_dates.txt'} & members.keys():
            raise ValueError(f'{path}: missing GTFS tables: {sorted(missing)} or calendar')
        for table, required_columns in REQUIRED_COLUMNS.items():
            with z.open(members[table]) as binary:
                with io.TextIOWrapper(binary, encoding='utf-8-sig') as stream:
                    reader = csv.DictReader(stream)
                    columns = {str(k).strip() for k in (reader.fieldnames or [])}
                    if required_columns - columns:
                        raise ValueError(f'{path}: {table} missing columns {sorted(required_columns-columns)}')
                    if next(reader, None) is None:
                        raise ValueError(f'{path}: {table} is empty')
        routes = rows(z,members,'routes.txt')
        agencies = rows(z,members,'agency.txt')
        cal = rows(z,members,'calendar.txt')
        exc = rows(z,members,'calendar_dates.txt')
        info = rows(z,members,'feed_info.txt')
        trips = rows(z,members,'trips.txt')
        if not routes or not trips or not agencies:
            raise ValueError(f'{path}: empty required table')
        route_types = collections.Counter(r['route_type'] for r in routes)
        service_trips = collections.Counter(r['service_id'] for r in trips)
        starts = [r['start_date'] for r in cal] + [r['date'] for r in exc if r['exception_type']=='1']
        ends = [r['end_date'] for r in cal] + [r['date'] for r in exc if r['exception_type']=='1']
        valid_tuesdays = []
        if starts:
            first,last = parse_date(min(starts)),parse_date(max(ends))
            day = first + dt.timedelta(days=(1-first.weekday())%7)
            while day <= last:
                count = sum(service_trips[s] for s in active_services(cal,exc,day))
                if count: valid_tuesdays.append(day.isoformat())
                day += dt.timedelta(days=7)
        effective = sum(service_trips[s] for s in active_services(cal,exc,target))
        date_ranges = dict(calendar_start=min((r['start_date'] for r in cal),default=None),
                           calendar_end=max((r['end_date'] for r in cal),default=None),
                           exception_start=min((r['date'] for r in exc),default=None),
                           exception_end=max((r['date'] for r in exc),default=None),
                           feed_info=[{k:v for k,v in r.items() if k in ('feed_start_date','feed_end_date','feed_version')} for r in info],
                           service_start=min(starts,default=None),service_end=max(ends,default=None))
        result = dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size,
                      retrieved_at_utc=dt.datetime.fromtimestamp(path.stat().st_mtime,dt.timezone.utc).isoformat(),
                      retrieved_at_basis='Local download file modification time',
                      tables=sorted(members),route_type_counts=dict(route_types),route_count=len(routes),trip_count=len(trips),
                      agency_timezones=sorted({r.get('agency_timezone','') for r in agencies}),agencies=agencies,
                      date_ranges=date_ranges,valid_tuesdays=valid_tuesdays,active_trip_count_target_date=effective,
                      target_date=target.isoformat(),validation='Required tables, nonempty agency/routes/trips, parsed calendars; not a formal GTFS validator')
    return result


def download_feed(spec, refresh=False):
    path = ROOT / spec['path']
    if path.exists() and not refresh:
        return path
    path.parent.mkdir(parents=True,exist_ok=True)
    request = urllib.request.Request(spec['download_url'],headers={'User-Agent':'APMA4903 academic transit research'})
    with urllib.request.urlopen(request,timeout=180) as response:
        data = response.read()
    if spec.get('archive_member'):
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            data = archive.read(spec['archive_member'])
    tmp = path.with_suffix('.download')
    tmp.write_bytes(data)
    try:
        inspect_feed(tmp)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


def build_manifest(download=False, refresh=False):
    cities=[]
    for spec in SOURCES:
        city={k:v for k,v in spec.items() if k!='feeds'}
        city['feeds']=[]
        for f in spec['feeds']:
            path=ROOT/f['path']
            if download: download_feed(f,refresh=refresh)
            if not path.exists():
                city['missing_agencies']=city['missing_agencies']+[f['agency']+' (file not downloaded)']
                continue
            entry={**f,**inspect_feed(path)}
            if not entry['active_trip_count_target_date']:
                entry['notes']=entry['notes']+['No scheduled trips on target date; use an explicitly valid service date.']
            if f['feed_id']=='bart':
                entry['notes']=entry['notes']+['feed_info ends 2026-08-30 although actual calendar services run through 2027-01-10; metadata conflict is retained.']
            if f.get('mirror_source_url'):
                entry['notes']=entry['notes']+['Agency-produced static feed downloaded from the named public archive, not the producer API.']
            city['feeds'].append(entry)
        calendars=[set(f['valid_tuesdays']) for f in city['feeds']]
        common=set.intersection(*calendars) if calendars else set()
        city['recommended_service_date']=(min(common,key=lambda s:abs((dt.date.fromisoformat(s)-TARGET_DATE).days)) if common else None)
        city['common_valid_tuesday_exists']=bool(common)
        city['downloaded_feed_count']=len(city['feeds'])
        city['has_bus']=any(any(k in f['route_type_counts'] for k in ('3','11')) for f in city['feeds'])
        city['has_rail']=any(any(k in f['route_type_counts'] for k in ('0','1','2','5','12')) for f in city['feeds'])
        cities.append(city)
    result=dict(generated_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),target_date=TARGET_DATE.isoformat(),
                policy='Public static GTFS only. No Google Routes API, realtime data, authentication credentials or paid services.',
                date_policy='Use common valid Tuesday nearest target where possible; dates come from active services linked to trips. Never extend service calendars.',
                cities=cities)
    MANIFEST.parent.mkdir(parents=True,exist_ok=True)
    temp=MANIFEST.with_suffix('.tmp')
    temp.write_text(json.dumps(result,indent=2)+'\n')
    temp.replace(MANIFEST)
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download',action='store_true',help='Download missing static schedules from listed sources')
    parser.add_argument('--refresh',action='store_true',help='With --download, replace cached snapshots')
    args=parser.parse_args()
    m=build_manifest(download=args.download,refresh=args.refresh)
    for city in m['cities']:
        print(city['city_id'],city['downloaded_feed_count'],'feeds, bus=',city['has_bus'],'rail=',city['has_rail'],'date=',city['recommended_service_date'])
