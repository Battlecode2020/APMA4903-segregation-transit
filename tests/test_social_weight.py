import math
import unittest
from unittest.mock import patch

from scripts.social_weight import MissingPriceError, RoutingError, social_weight


class SocialWeightTests(unittest.TestCase):
    def calculate(self, **kwargs):
        options = dict(beta=.2, omega=12, c0=1, api_key="test-key")
        options.update(kwargs)
        return social_weight((40.8, -73.96), (40.75, -73.98), **options)

    @patch("scripts.social_weight._post")
    def test_units_mode_choice_and_payload(self, post):
        post.side_effect = [
            {"routes": [{"duration": "3600s", "distanceMeters": 4000}]},
            {"routes": [{"duration": "600s", "distanceMeters": 5000,
                         "travelAdvisory": {"transitFare": {"currencyCode": "USD", "units": "2", "nanos": 500000000}}}]},
        ]
        result = self.calculate()
        self.assertAlmostEqual(result.cost_usd, 4.5)
        self.assertAlmostEqual(result.f, math.exp(-.9))
        self.assertEqual(result.best_mode, "TRANSIT")
        self.assertEqual(post.call_args_list[0].args[0]["origin"]["location"]["latLng"]["latitude"], 40.8)
        self.assertIn("routes.travelAdvisory.transitFare", post.call_args.args[2])
        self.assertEqual(post.call_args.args[0]["departureTime"], result.departure_time)

    @patch("scripts.social_weight._post")
    def test_local_floor_without_api(self, post):
        result = social_weight((40, -74), (40, -74), beta=.2, omega=12, c0=1)
        self.assertAlmostEqual(result.f, math.exp(-.2))
        post.assert_not_called()

    @patch("scripts.social_weight._post", return_value={"routes": [{"duration": "1s"}]})
    def test_floor(self, post):
        self.assertEqual(self.calculate(modes=["WALK"]).cost_usd, 1)

    @patch("scripts.social_weight._post", return_value={"routes": [{"duration": "600s"}]})
    def test_missing_fare_is_not_free(self, post):
        with self.assertRaises(MissingPriceError):
            self.calculate(modes=["TRANSIT"])
        self.assertEqual(self.calculate(modes=["TRANSIT"], transit_fare_usd=3).cost_usd, 5)

    @patch("scripts.social_weight._post")
    def test_driving_costs(self, post):
        post.return_value = {"routes": [{"duration": "1800s", "distanceMeters": 10000,
                            "travelAdvisory": {"tollInfo": {"estimatedPrice": [{"currencyCode": "USD", "units": "3"}]}}}]}
        result = self.calculate(modes=["DRIVE"], driving_usd_per_km=.2, parking_usd=4)
        self.assertEqual(result.cost_usd, 15)
        post.return_value["routes"][0]["travelAdvisory"]["tollInfo"] = {}
        with self.assertRaises(MissingPriceError):
            self.calculate(modes=["DRIVE"], driving_usd_per_km=.2)

    @patch("scripts.social_weight._post", return_value={})
    def test_unreachable(self, post):
        result = self.calculate()
        self.assertEqual(result.f, 0)
        self.assertEqual(result.cost_usd, math.inf)

    @patch("scripts.social_weight._post", side_effect=RoutingError("HTTP 403"))
    def test_api_failure_is_not_unreachable(self, post):
        with self.assertRaises(RoutingError):
            self.calculate()

    @patch("scripts.social_weight._post")
    def test_foreign_currency(self, post):
        post.return_value = {"routes": [{"duration": "600s", "travelAdvisory": {
            "transitFare": {"currencyCode": "EUR", "units": "2"}}}]}
        with self.assertRaises(MissingPriceError):
            self.calculate(modes=["TRANSIT"])

    def test_validation(self):
        for changes in ({"beta": -1}, {"omega": math.nan}, {"c0": 0},
                        {"modes": []}, {"modes": ["DRIVE"]}):
            with self.assertRaises(ValueError):
                self.calculate(**changes)


if __name__ == "__main__":
    unittest.main()
