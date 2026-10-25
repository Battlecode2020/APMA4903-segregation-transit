"""Local fixture tests for GTFS source validation and calendar provenance."""
import datetime as dt
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from multicity_sources import active_services, inspect_feed, table_members


class SourceTests(unittest.TestCase):
    def fixture(self, path, *, calendar=True, bad_stop_times=False, spaced=False):
        tables = {
            'agency.txt':'agency_name,agency_url,agency_timezone\nTest Transit,https://example.org,America/New_York\n',
            'routes.txt':'route_id,route_type\nr,3\n',
            'stops.txt':'stop_id,stop_lat,stop_lon\ns,40,-73\n',
            'trips.txt':'route_id,service_id,trip_id\nr,weekday,t\n',
            'stop_times.txt':'trip_id,arrival_time,departure_time,stop_id,stop_sequence\nt,08:00:00,08:00:00,s,1\n',
            'calendar_dates.txt':'service_id,date,exception_type\nweekday,20261020,2\nweekday,20261027,1\n',
        }
        if calendar:
            tables['calendar.txt']='service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date\nweekday,1,1,1,1,1,0,0,20261019,20261101\n'
        if bad_stop_times:
            tables['stop_times.txt']='trip_id,stop_id\nt,s\n'
        if spaced:
            tables={k:', '.join(v.split(',')) for k,v in tables.items()}
        with zipfile.ZipFile(path,'w') as z:
            for name,data in tables.items():z.writestr('feed/'+name,data)

    def test_exceptions_override_weekly_calendar(self):
        cal=[dict(service_id='a',start_date='20261001',end_date='20261031',tuesday='1')]
        exc=[dict(service_id='a',date='20261020',exception_type='2'),dict(service_id='b',date='20261020',exception_type='1')]
        self.assertEqual(active_services(cal,exc,dt.date(2026,10,20)),{'b'})

    def test_active_tuesdays_are_not_fabricated(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'feed.zip';self.fixture(path)
            result=inspect_feed(path)
            self.assertEqual(result['valid_tuesdays'],['2026-10-27'])
            self.assertEqual(result['active_trip_count_target_date'],0)
            self.assertEqual(result['route_type_counts'],{'3':1})
            self.assertEqual(len(result['sha256']),64)
            self.assertEqual(result['agency_timezones'],['America/New_York'])

    def test_calendar_dates_only_is_supported(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'feed.zip';self.fixture(path,calendar=False)
            self.assertEqual(inspect_feed(path)['valid_tuesdays'],['2026-10-27'])

    def test_metra_style_whitespace_is_supported(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'feed.zip';self.fixture(path,spaced=True)
            self.assertEqual(inspect_feed(path)['valid_tuesdays'],['2026-10-27'])

    def test_invalid_required_columns_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'feed.zip';self.fixture(path,bad_stop_times=True)
            with self.assertRaisesRegex(ValueError,'stop_times.txt missing columns'):
                inspect_feed(path)

    def test_ambiguous_nested_tables_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'feed.zip'
            with zipfile.ZipFile(path,'w') as z:
                z.writestr('a/routes.txt','route_id,route_type\nr,3\n')
                z.writestr('b/routes.txt','route_id,route_type\nr,3\n')
            with zipfile.ZipFile(path) as z:
                with self.assertRaisesRegex(ValueError,'Duplicate GTFS table'):
                    table_members(z)

if __name__=='__main__':unittest.main()
