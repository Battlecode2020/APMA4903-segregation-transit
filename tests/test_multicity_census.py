"""City-limit aggregation tests: boundary blocks, zero population, exact totals."""
import sys
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import multicity_census as census


class CityCensusTests(unittest.TestCase):
    def fixture(self):
        def row(level, geoid, pop, counts, place="12345", lat="40", lon="-75"):
            return {"SUMLEV": level, "GEOCODE": geoid, "PLACE": place,
                    "NAME": "Test city", "population": pop,
                    **dict(zip(census.RACE_NAMES, counts)), "INTPTLAT": lat, "INTPTLON": lon}
        return pd.DataFrame([
            row("160", "4212345", 30, [12, 8, 15, 3]),
            row("750", "420010001001000", 10, [4, 3, 5, 1], lon="-75.01"),
            row("750", "420010001001001", 20, [8, 5, 10, 2], lon="-75.00"),
            row("750", "420010002001000", 0, [0, 0, 0, 0]),
            row("750", "420010001001002", 70, [60, 5, 8, 3], place="99999"),
            row("140", "42001000100", 100, [72, 13, 23, 6], place=""),
            row("140", "42001000200", 0, [0, 0, 0, 0], place=""),
        ])

    def aggregate(self, records=None):
        with patch.dict(census.CITIES, {"test": ("Test", "42", "12345", 30)}):
            return census.aggregate_city(self.fixture() if records is None else records, "test")

    def test_exact_place_membership_excludes_neighbors_in_same_tract(self):
        nodes, _, blocks, manifest = self.aggregate()
        self.assertEqual(nodes.population.tolist(), [30])
        self.assertEqual(nodes.full_tract_population.tolist(), [100])
        self.assertEqual(nodes.city_share_of_tract_population.tolist(), [.3])
        self.assertEqual(nodes[census.RACE_NAMES].iloc[0].tolist(), [12, 8, 15, 3])
        self.assertEqual(set(blocks.PLACE), {"12345"})
        self.assertEqual(manifest["tracts_with_population_outside_city"], 1)

    def test_zero_population_tract_portions_excluded_without_losing_counts(self):
        nodes, zero, _, manifest = self.aggregate()
        self.assertEqual(len(nodes), 1)
        self.assertEqual(zero.tract_geoid.tolist(), ["42001000200"])
        self.assertEqual(manifest["zero_population_portions_excluded"], 1)
        self.assertEqual(nodes.matrix_index.tolist(), [0])

    def test_population_weighted_centroid_and_populated_block_representative(self):
        nodes, _, _, _ = self.aggregate()
        self.assertAlmostEqual(nodes.population_centroid_longitude.iloc[0], -75.0033333333, places=6)
        self.assertEqual(nodes.representative_block_geoid.iloc[0], "420010001001001")
        self.assertEqual(nodes.longitude.iloc[0], -75)

    def test_hispanic_is_not_made_mutually_exclusive(self):
        nodes, _, _, manifest = self.aggregate()
        self.assertGreater(nodes[census.RACE_NAMES].sum(axis=1).iloc[0], nodes.population.iloc[0])
        self.assertEqual(manifest["race_population_shares"]["hispanic_latino_any_race"], .5)

    def test_mismatch_against_independently_published_place_rejected(self):
        records = self.fixture()
        records.loc[0, "black_alone"] += 1
        with self.assertRaisesRegex(ValueError, "Block sum differs"):
            self.aggregate(records)

    def test_duplicate_block_rejected(self):
        records = self.fixture()
        records.loc[2, "GEOCODE"] = records.loc[1, "GEOCODE"]
        with self.assertRaisesRegex(ValueError, "Block GEOIDs"):
            self.aggregate(records)


if __name__ == "__main__":
    unittest.main()
