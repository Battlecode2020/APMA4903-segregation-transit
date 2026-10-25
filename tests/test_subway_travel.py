from datetime import date
import unittest
import numpy as np
import pandas as pd
from scripts.subway_travel import active_services, scan_connections, seconds


class SubwayScheduleTests(unittest.TestCase):
    def test_gtfs_after_midnight(self):
        self.assertEqual(seconds('25:10:00'),90600)
        with self.assertRaises(ValueError): seconds('12:61:00')

    def test_calendar_exceptions_override_weekday(self):
        calendar=pd.DataFrame([dict(service_id='weekday',start_date='20260101',end_date='20261231',monday='1')])
        exceptions=pd.DataFrame([dict(service_id='weekday',date='20261019',exception_type='2'),
                                 dict(service_id='holiday',date='20261019',exception_type='1')])
        self.assertEqual(active_services(calendar,exceptions,date(2026,10,19)),{'holiday'})

    def test_staying_onboard_preserves_dwell_without_transfer_penalty(self):
        connections=[(100,200,0,1,'train','A',1),(220,300,1,2,'train','A',2)]
        result=scan_connections(connections,{0:[(0,180)],1:[(1,180)],2:[(2,180)]},[50,np.inf,np.inf],0)
        label=result[2]
        self.assertEqual(label.time,300)
        self.assertEqual(label.access,50)
        self.assertEqual(label.wait,50)
        self.assertEqual(label.ride,200)  # includes 20 seconds dwelling at stop 1
        self.assertEqual(label.transfer,0)

    def test_transfer_buffer_prevents_impossible_connection(self):
        connections=[(100,200,0,1,'A1','A',1),(250,300,1,2,'B1','B',1),(400,500,1,2,'B2','B',1)]
        result=scan_connections(connections,{1:[(1,180)]},[0,np.inf,np.inf],0)
        self.assertEqual(result[2].time,500)
        self.assertEqual(result[2].transfer,180)
        self.assertEqual(result[2].wait,120)
        self.assertEqual(result[2].access+result[2].ride+result[2].wait+result[2].transfer,500)

    def test_later_express_can_beat_earlier_local(self):
        connections=[(100,900,0,1,'local','L',1),(200,500,0,1,'express','X',1)]
        result=scan_connections(connections,{1:[(1,120)]},[0,np.inf],0)
        self.assertEqual(result[1].time,500)

    def test_missed_train_and_disconnected_origin_do_not_create_routes(self):
        connections=[(100,200,0,1,'A','A',1)]
        self.assertIsNone(scan_connections(connections,{},[101,np.inf],0)[1])
        self.assertIsNone(scan_connections(connections,{},[np.inf,0],0)[0])

    def test_transfer_between_platforms(self):
        connections=[(100,200,0,1,'A','A',1),(350,450,2,3,'B','B',1)]
        result=scan_connections(connections,{1:[(2,120)]},[0,np.inf,np.inf,np.inf],0)
        self.assertEqual(result[3].time,450)
        self.assertEqual(result[3].transfer,120)
        self.assertEqual(result[3].wait,130)
