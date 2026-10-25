import json
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile

import numpy as np
import pandas as pd
from shapely.geometry import Polygon, box

from scripts.multicity_transit import (
    TransitParameters, active_services, choose_service_dates, compute_city,
    distance_km, gtfs_seconds, load_patterns, route_mode,
)


class MulticityTransitTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.params = TransitParameters(bus_access_km=.2, rail_access_km=.2,
                                        transfer_walk_km=0, dijkstra_batch_size=2)

    def tearDown(self):
        self.temporary.cleanup()

    def feed(self, *, frequency=False, restricted=False, branches=False):
        stops = pd.DataFrame({"stop_id": ["a", "b", "c", "d", "e"],
                              "stop_lat": [40]*5, "stop_lon": [-75, -74.96, -74.92, -75.04, -74.88]})
        trips, times = [], []
        patterns = [("r", ["a", "b", "c"])]
        if branches:
            patterns.append(("r", ["d", "b", "e"]))
        for p, (route, sequence) in enumerate(patterns):
            for hour in ([0] if frequency else [7, 8, 9]):
                trip = f"p{p}h{hour}"
                trips.append({"route_id": route, "service_id": "weekday", "trip_id": trip})
                for k, stop in enumerate(sequence):
                    second = hour*3600+k*600+(30 if k > 1 else 0)
                    fmt = lambda t: f"{t//3600:02}:{t%3600//60:02}:{t%60:02}"
                    times.append({"trip_id": trip, "stop_id": stop, "stop_sequence": k,
                                  "arrival_time": fmt(second), "departure_time": fmt(second+(30 if k == 1 else 0)),
                                  "pickup_type": "1" if restricted and stop == "b" else "0",
                                  "drop_off_type": "1" if restricted and stop == "b" else "0"})
        tables = {
            "stops": stops,
            "routes": pd.DataFrame({"route_id": ["r"], "route_type": ["3"]}),
            "trips": pd.DataFrame(trips), "stop_times": pd.DataFrame(times),
            "agency": pd.DataFrame({"agency_timezone": ["America/New_York"]}),
            "calendar": pd.DataFrame([dict(service_id="weekday", monday=1, tuesday=1, wednesday=1,
                thursday=1, friday=1, saturday=0, sunday=0, start_date="20261001", end_date="20261031")]),
        }
        if frequency:
            tables["frequencies"] = pd.DataFrame({"trip_id": ["p0h0"], "start_time": ["07:00:00"],
                "end_time": ["10:00:00"], "headway_secs": [600], "exact_times": [0]})
        path = self.root/"feed.zip"
        with ZipFile(path, "w") as archive:
            for name, frame in tables.items(): archive.writestr(name+".txt", frame.to_csv(index=False))
        return path, stops

    def nodes(self, stops):
        frame = stops.rename(columns={"stop_id": "tract_geoid", "stop_lat": "latitude", "stop_lon": "longitude"}).copy()
        frame["matrix_index"] = np.arange(len(frame))
        return frame

    def run_city(self, feed, stops, **kwargs):
        compute_city(self.nodes(stops), [feed], self.root/"output", parameters=self.params, **kwargs)
        with np.load(self.root/"output/modes.npz") as data:
            return {key: data[key].copy() for key in data.files}

    def test_directed_rides_include_wait_dwell_and_entrances(self):
        feed, stops = self.feed()
        data = self.run_city(feed, stops)
        # 60 access + 1,800 expected wait + 600 ride + 30 dwell +
        # 600 ride + 60 egress; no repeated wait when remaining onboard.
        self.assertAlmostEqual(data["transit_seconds"][0, 2], 3150)
        self.assertTrue(np.isinf(data["transit_seconds"][2, 0]))
        self.assertTrue(np.isinf(np.diag(data["transit_seconds"])).all())
        self.assertFalse(data["walk_barrier_check_available"].item())

    def test_no_boarding_or_alighting_at_restricted_stop(self):
        feed, stops = self.feed(restricted=True)
        data = self.run_city(feed, stops)
        self.assertTrue(np.isfinite(data["transit_seconds"][0, 2]))
        self.assertTrue(np.isinf(data["transit_seconds"][1, 2]))
        self.assertTrue(np.isinf(data["transit_seconds"][0, 1]))

    def test_route_branches_require_an_actual_transfer(self):
        feed, stops = self.feed(branches=True)
        data = self.run_city(feed, stops)
        # Separate exact stop patterns cannot jump branches without reboarding.
        # B boarding wait differs slightly because its 9:10 departure is still
        # inside the window; all three departures contribute.
        self.assertAlmostEqual(data["transit_seconds"][0, 4], 4980)

    def test_frequency_templates_are_not_counted_as_single_scheduled_trip(self):
        feed, stops = self.feed(frequency=True)
        data = self.run_city(feed, stops)
        self.assertAlmostEqual(data["transit_seconds"][0, 2], 1650)
        manifest = json.loads((self.root/"output/manifest.json").read_text())
        self.assertEqual(manifest["feeds"][0]["diagnostics"]["frequency_trip_templates"], 1)

    def test_restrictive_minimum_cannot_be_bypassed_by_other_transfer(self):
        feed, stops = self.feed(branches=True)
        # The implementation conservatively extends a B->D minimum to every
        # onward transfer after alighting at B, avoiding physical-stop bypasses.
        transfers = pd.DataFrame([{"from_stop_id": "b", "to_stop_id": "d",
                                    "from_route_id": "r", "transfer_type": "2",
                                    "min_transfer_time": "600"}])
        with ZipFile(feed, "a") as archive:
            archive.writestr("transfers.txt", transfers.to_csv(index=False))
        data = self.run_city(feed, stops)
        self.assertAlmostEqual(data["transit_seconds"][0, 4], 5520)
        self.assertAlmostEqual(data["transit_seconds"][0, 2], 3150)
        manifest = json.loads((self.root/"output/manifest.json").read_text())
        self.assertEqual(manifest["transfer_rule_diagnostics"]["minimums_conservatively_applied_to_all_onward_transfers"], 1)

    def test_transfer_prohibition_preserves_direct_egress_and_initial_boarding(self):
        feed, stops = self.feed(branches=True)
        transfers = pd.DataFrame([{"from_stop_id": "b", "to_stop_id": "d", "transfer_type": "3"}])
        with ZipFile(feed, "a") as archive:
            archive.writestr("transfers.txt", transfers.to_csv(index=False))
        data = self.run_city(feed, stops)
        self.assertTrue(np.isinf(data["transit_seconds"][0, 4]))
        self.assertTrue(np.isfinite(data["transit_seconds"][0, 1]))
        self.assertTrue(np.isfinite(data["transit_seconds"][1, 4]))

    def test_water_crossing_is_flagged_and_access_filtered(self):
        feed, stops = self.feed()
        water = box(-74.99, 39.9, -74.98, 40.1)
        data = self.run_city(feed, stops, water_geometry=water)
        # GTFS bus itself may cross a bridge. Direct walking is just a distance
        # proxy; a strict pedestrian chord crossing that water is unavailable.
        self.assertTrue(data["walk_water_crossing"][0, 2])
        self.assertTrue(np.isinf(data["walk_barrier_checked_seconds"][0, 2]))
        self.assertTrue(np.isfinite(data["walk_seconds"][0, 2]))
        self.assertTrue(np.isfinite(data["transit_seconds"][0, 2]))
        shifted = stops.copy()
        shifted.loc[0, "stop_lon"] = -74.985
        # A representative inside water cannot use a transit access chord.
        compute_city(self.nodes(shifted), [feed], self.root/"water", parameters=self.params, water_geometry=water)
        with np.load(self.root/"water/modes.npz") as result:
            self.assertEqual(result["access_stop_count"][0], 0)

    def test_empty_feed_has_no_invented_transit(self):
        _, stops = self.feed()
        manifest = compute_city(self.nodes(stops), [], self.root/"walk", parameters=self.params)
        with np.load(self.root/"walk/modes.npz") as data:
            self.assertTrue(np.isinf(data["transit_seconds"]).all())
            self.assertTrue(np.isfinite(data["walk_seconds"]).all())
        self.assertEqual(manifest["transit_reachable_nonlocal_pairs"], 0)

    def test_empty_land_fails_before_caching_zero_access_results(self):
        feed, stops = self.feed()
        destination = self.root/"empty_land"
        with self.assertRaisesRegex(ValueError, "land geometry is empty"):
            compute_city(self.nodes(stops), [feed], destination,
                         parameters=replace(self.params, access_policy="uncapped"),
                         land_geometry=Polygon())
        self.assertFalse((destination/"modes.npz").exists())
        self.assertFalse((destination/"manifest.json").exists())

    def test_invalid_land_or_water_fails_and_empty_water_is_legitimate(self):
        feed, stops = self.feed()
        bowtie = Polygon([(-75, 40), (-74, 41), (-74, 40), (-75, 41), (-75, 40)])
        self.assertFalse(bowtie.is_valid)
        for kind in ("land", "water"):
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, f"{kind} geometry is invalid"):
                compute_city(self.nodes(stops), [feed], self.root/kind,
                             parameters=self.params, **{kind+"_geometry": bowtie})
        reference = self.run_city(feed, stops)
        compute_city(self.nodes(stops), [feed], self.root/"empty_water",
                     parameters=self.params, water_geometry=Polygon())
        with np.load(self.root/"empty_water/modes.npz") as result:
            np.testing.assert_allclose(result["transit_seconds"], reference["transit_seconds"])

    def test_calendar_exception_and_extended_time(self):
        feed, _ = self.feed()
        self.assertEqual(choose_service_dates([feed])[0].isoformat(), "2026-10-20")
        with ZipFile(feed) as archive:
            cal = pd.read_csv(archive.open("calendar.txt"), dtype=str)
        exceptions = pd.DataFrame({"service_id": ["weekday", "special"], "date": ["20261020"]*2,
                                   "exception_type": ["2", "1"]})
        from datetime import date
        self.assertEqual(active_services(cal, exceptions, date(2026, 10, 20)), {"special"})
        self.assertEqual(gtfs_seconds("25:03:02"), 90182)
        self.assertEqual(route_mode("11"), "bus")
        self.assertEqual(route_mode("12"), "rail")
        self.assertIsNone(route_mode("4"))

    def test_uncapped_long_access_and_egress_are_fully_charged(self):
        feed, stops = self.feed()
        nodes = pd.DataFrame({"matrix_index": [0, 1], "tract_geoid": ["west", "east"],
                              "latitude": [40, 40], "longitude": [-75.02, -74.90]})
        compute_city(nodes, [feed], self.root/"capped", parameters=self.params)
        with np.load(self.root/"capped/modes.npz") as capped:
            self.assertTrue(np.isinf(capped["transit_seconds"][0, 1]))
        params = replace(self.params, access_policy="uncapped")
        report = compute_city(nodes, [feed], self.root/"uncapped", parameters=params)
        with np.load(self.root/"uncapped/modes.npz") as data:
            walk = distance_km(nodes[["latitude", "longitude"]], stops[["stop_lat", "stop_lon"]])*params.walking_detour*3600/params.walking_speed_kmh
            access, egress = walk[0, 0]+60, walk[1, 2]+60
            self.assertAlmostEqual(data["transit_seconds"][0, 1], access+1800+600+30+600+egress, places=6)
            self.assertAlmostEqual(data["transit_access_seconds"][0, 1], access, places=6)
            self.assertAlmostEqual(data["transit_egress_seconds"][0, 1], egress, places=6)
            self.assertTrue(data["transit_access_exceeds_legacy_radius"][0, 1])
            self.assertTrue(data["transit_egress_exceeds_legacy_radius"][0, 1])
            self.assertEqual(data["transit_stop_ids"][data["transit_first_boarding_stop_index"][0, 1]], "0:a")
            self.assertEqual(data["transit_stop_ids"][data["transit_last_alighting_stop_index"][0, 1]], "0:c")
        self.assertEqual(report["routing_diagnostics"]["maximum_temporary_origins"], 2)

    def test_uncapped_matches_dense_all_stop_reference_and_preserves_transfers(self):
        feed, stops = self.feed(branches=True)
        # An enormous capped radius is a small-fixture reference that explicitly
        # stores every access edge. The production uncapped algorithm batches.
        huge = replace(self.params, bus_access_km=1000, rail_access_km=1000)
        compute_city(self.nodes(stops), [feed], self.root/"dense", parameters=huge)
        transfers = pd.DataFrame([{"from_stop_id": "b", "to_stop_id": "d", "transfer_type": "2", "min_transfer_time": 600}])
        with ZipFile(feed, "a") as archive:
            archive.writestr("transfers.txt", transfers.to_csv(index=False))
        compute_city(self.nodes(stops), [feed], self.root/"dense_transfer", parameters=huge)
        for batch_size in (1, 3):
            params = replace(self.params, access_policy="uncapped", dijkstra_batch_size=batch_size)
            compute_city(self.nodes(stops), [feed], self.root/f"batch{batch_size}", parameters=params)
            with np.load(self.root/"dense_transfer/modes.npz") as reference, np.load(self.root/f"batch{batch_size}/modes.npz") as actual:
                np.testing.assert_allclose(actual["transit_seconds"], reference["transit_seconds"], atol=1e-8)
                np.testing.assert_allclose(actual["walk_seconds"], reference["walk_seconds"], atol=1e-8)

    def test_uncapped_transit_still_requires_a_real_ride(self):
        feed, _ = self.feed()
        nodes = pd.DataFrame({"matrix_index": [0, 1], "tract_geoid": ["same-a", "same-b"],
                              "latitude": [40, 40], "longitude": [-75, -75]})
        params = replace(self.params, access_policy="uncapped")
        compute_city(nodes, [feed], self.root/"ride", parameters=params)
        with np.load(self.root/"ride/modes.npz") as data:
            self.assertEqual(data["walk_seconds"][0, 1], 0)
            self.assertGreater(data["transit_seconds"][0, 1], 1800+600)
            self.assertNotEqual(data["transit_first_boarding_stop_index"][0, 1], data["transit_last_alighting_stop_index"][0, 1])
        compute_city(nodes, [], self.root/"no_feed", parameters=params)
        with np.load(self.root/"no_feed/modes.npz") as data:
            self.assertTrue(np.isinf(data["transit_seconds"]).all())
            self.assertTrue((data["transit_first_boarding_stop_index"] == -1).all())

    def test_uncapped_water_rejection_and_optional_flags(self):
        feed, stops = self.feed()
        water = box(-75.03, 39.9, -75.01, 40.1)
        nodes = pd.DataFrame({"matrix_index": [0, 1], "tract_geoid": ["west", "east"],
                              "latitude": [40, 40], "longitude": [-75.04, -74.92]})
        params = replace(self.params, access_policy="uncapped")
        compute_city(nodes, [feed], self.root/"reject", parameters=params, water_geometry=water)
        with np.load(self.root/"reject/modes.npz") as data:
            self.assertTrue(np.isinf(data["transit_seconds"][0, 1]))
            self.assertEqual(data["access_stop_count"][0], 0)
        compute_city(nodes, [feed], self.root/"flag", parameters=replace(params, pedestrian_barrier_policy="flag_only"), water_geometry=water)
        with np.load(self.root/"flag/modes.npz") as data:
            self.assertTrue(np.isfinite(data["transit_seconds"][0, 1]))
            self.assertTrue(data["transit_access_barrier_flag"][0, 1])
            self.assertFalse(data["transit_egress_barrier_flag"][0, 1])

    def test_uncapped_cannot_increase_cost_and_walking_can_still_win(self):
        from scripts.multicity_analysis import generalized_costs
        feed, stops = self.feed(branches=True)
        capped = self.run_city(feed, stops)
        compute_city(self.nodes(stops), [feed], self.root/"unlimited", parameters=replace(self.params, access_policy="uncapped"))
        with np.load(self.root/"unlimited/modes.npz") as actual:
            finite = np.isfinite(capped["transit_seconds"])
            self.assertTrue((actual["transit_seconds"][finite] <= capped["transit_seconds"][finite]+1e-8).all())
            before = generalized_costs(capped["walk_seconds"], capped["transit_seconds"])
            after = generalized_costs(actual["walk_seconds"], actual["transit_seconds"])
            self.assertTrue((after["C_usd"] <= before["C_usd"]+1e-8).all())
            # Reverse-direction candidate now exists via long walking access,
            # but taking that detour cannot beat direct walking in this fixture.
            self.assertTrue(np.isfinite(actual["transit_seconds"][2, 0]))
            self.assertEqual(after["chosen_mode"][2, 0], "WALK")


if __name__ == "__main__":
    unittest.main()
