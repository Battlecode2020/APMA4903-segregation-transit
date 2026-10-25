import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts.nyc_routes import RouteConfig, collect, export_matrices, open_cache, plan, population_transition


class RouteMatrixTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.zones = pd.DataFrame({'matrix_index':[0,1], 'zone_id':['A','B'],
            'latitude':[40.8,40.7], 'longitude':[-73.9,-74.0], 'TOTPOP':[100,200]})
        self.config = RouteConfig((datetime.now(timezone.utc)+timedelta(days=1)).isoformat())
        self.cache = self.root/'responses.sqlite'
        self.reply = {'routes':[{'duration':'600s', 'distanceMeters':2000,
            'travelAdvisory':{'transitFare':{'currencyCode':'USD','units':'3'}}}]}

    def collect(self, **kwargs):
        return collect(self.zones,self.config,self.cache,allow_api=True,max_requests=kwargs.pop('max_requests',2),
                       api_key='fake-test-key',interval_seconds=0,**kwargs)

    def export(self, **kwargs):
        export_matrices(self.zones,self.config,self.cache,self.root/'export',**kwargs)
        return np.load(self.root/'export/matrices.npz')

    @patch('scripts.nyc_routes._post')
    @patch('scripts.social_weight._post')
    def test_dry_run_and_export_never_network(self, legacy_post, post):
        result = collect(self.zones,self.config,self.cache)
        self.assertEqual(result['planned_requests'],2)
        self.assertFalse(self.cache.exists())
        data = self.export()
        self.assertEqual(data['status'][0,1],'pending')
        self.assertTrue(np.isnan(data['f'][0,1]))
        self.assertAlmostEqual(data['f'][0,0],math.exp(-.125))
        post.assert_not_called()
        legacy_post.assert_not_called()

    @patch('scripts.nyc_routes._post')
    def test_resume_direction_units_and_cap(self, post):
        post.return_value = self.reply
        self.assertEqual(self.collect(max_requests=1)['requests_this_run'],1)
        data = self.export()
        self.assertEqual(data['status'][0,1],'ok')
        self.assertEqual(data['status'][1,0],'pending')
        post.return_value = {'routes':[]}
        self.assertEqual(self.collect()['requests_this_run'],1)
        self.assertEqual(self.collect()['requests_this_run'],0)
        self.assertEqual(post.call_count,2)
        data = self.export()
        self.assertEqual(data['duration_seconds'][0,1],600)
        self.assertEqual(data['price_usd'][0,1],3)
        self.assertEqual(data['cost_usd'][0,1],5.5)
        self.assertEqual(data['f'][1,0],0)
        self.assertTrue(np.isinf(data['cost_usd'][1,0]))
        self.assertEqual(data['status'][1,0],'no_route')
        self.assertEqual(data['zone_ids'].tolist(),['A','B'])
        # Parameters can be changed without collecting again.
        self.assertEqual(self.export(omega=6)['cost_usd'][0,1],4)
        self.assertEqual(post.call_count,2)

    @patch('scripts.nyc_routes._post')
    def test_failure_not_unreachable_and_retry_is_explicit(self, post):
        post.side_effect = RuntimeError('test transport failure')
        with self.assertRaises(RuntimeError):
            self.collect()
        with self.assertRaises(RuntimeError):
            self.collect()
        self.assertEqual(post.call_count,1)
        data = self.export()
        self.assertEqual(data['status'][0,1],'request_error')
        self.assertTrue(np.isnan(data['f'][0,1]))
        post.side_effect = None
        post.return_value = self.reply
        self.collect(retry_errors=True)
        self.assertEqual(self.export()['status'][0,1],'ok')

    @patch('scripts.nyc_routes._post')
    def test_unknown_fare_retained_and_offline_override(self, post):
        post.return_value = {'routes':[{'duration':'600s','distanceMeters':2000}]}
        self.collect()
        data = self.export()
        self.assertEqual(data['status'][0,1],'price_unknown')
        self.assertEqual(data['fastest_duration_seconds_by_mode'][0,0,1],600)
        self.assertTrue(np.isnan(data['f'][0,1]))
        self.assertEqual(self.export(transit_fare_usd=4)['cost_usd'][0,1],6.5)
        self.assertEqual(post.call_count,2)

    @patch('scripts.nyc_routes._post')
    def test_cache_identity_prevents_mixing_experiments(self, post):
        post.return_value = self.reply
        self.collect()
        self.zones.loc[0,'latitude'] += .01
        with self.assertRaises(ValueError):
            self.collect()
        self.assertEqual(post.call_count,2)

    @patch('scripts.nyc_routes._post')
    def test_multimode_partial_does_not_fake_minimum(self, post):
        self.config = RouteConfig(self.config.departure_time,('WALK','TRANSIT'))
        self.assertEqual(plan(self.zones,self.config)['planned_requests'],4)
        post.return_value = {'routes':[{'duration':'3600s','distanceMeters':3000}]}
        self.collect(max_requests=1)
        self.assertEqual(self.export()['status'][0,1],'pending')

    @patch('scripts.nyc_routes._post')
    def test_opt_in_still_needs_positive_cap(self, post):
        with self.assertRaises(ValueError):
            collect(self.zones,self.config,self.cache,allow_api=True)
        post.assert_not_called()

    @patch('scripts.nyc_routes._post')
    def test_walking_ferry_not_free(self, post):
        self.config = RouteConfig(self.config.departure_time,('WALK',))
        post.return_value = {'routes':[{'duration':'3600s','legs':[{'steps':[
            {'navigationInstruction':{'maneuver':'FERRY'}}]}]}]}
        self.collect()
        self.assertEqual(self.export()['status'][0,1],'price_unknown')
        self.assertIn('navigationInstruction',post.call_args.args[2])

    def test_stable_population_transition(self):
        result = population_transition(np.array([[-10000,-10000],[-np.inf,-np.inf]]),[100,200])
        np.testing.assert_allclose(result,[[1/3,2/3],[0,0]])
        with self.assertRaises(ValueError):
            population_transition([[0,np.nan],[0,0]],[100,200])


if __name__ == '__main__':
    unittest.main()
