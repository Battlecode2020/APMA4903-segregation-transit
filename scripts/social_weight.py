"""Coordinate-based social weights using Google Routes (Python 3.11+)."""
from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ENDPOINT = "https://routes.googleapis.com/directions/v2:computeRoutes"
MODES = {"WALK", "BICYCLE", "DRIVE", "TRANSIT"}


class RoutingError(RuntimeError):
    """The API request failed; this does not mean the destination is unreachable."""


class MissingPriceError(ValueError):
    """A route exists, but its monetary cost cannot be determined."""


@dataclass(frozen=True)
class ModeCost:
    mode: str
    hours: float
    distance_meters: float
    price_usd: float
    generalized_cost_usd: float


@dataclass(frozen=True)
class SocialWeight:
    f: float
    log_f: float
    cost_usd: float
    best_mode: str | None
    alternatives: tuple[ModeCost, ...]
    unavailable_modes: tuple[str, ...]
    departure_time: str


def _nonnegative(name, value):
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")


def _point(coords):
    if len(coords) != 2:
        raise ValueError("Coordinates must be (latitude, longitude)")
    lat, lon = map(float, coords)
    if not (math.isfinite(lat) and math.isfinite(lon)
            and -90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError("Invalid latitude/longitude; use WGS84 degrees")
    return {"location": {"latLng": {"latitude": lat, "longitude": lon}}}


def _money(value):
    if value.get("currencyCode") != "USD":
        raise MissingPriceError("Expected USD; convert other currencies explicitly")
    amount = float(value.get("units", 0)) + float(value.get("nanos", 0)) / 1e9
    _nonnegative("API price", amount)
    return amount


def _post(body, api_key, field_mask, timeout):
    request = Request(ENDPOINT, data=json.dumps(body).encode(), headers={
        "Content-Type": "application/json", "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": field_mask,
    }, method="POST")
    try:
        with urlopen(request, timeout=timeout) as response:
            result = json.load(response)
    except HTTPError as exc:
        raise RoutingError(f"Google Routes HTTP {exc.code}; check credentials, quota and request") from None
    except (URLError, TimeoutError, ValueError) as exc:
        raise RoutingError(f"Google Routes transport/response failure: {type(exc).__name__}") from None
    if not isinstance(result, dict) or "error" in result:
        raise RoutingError("Google Routes returned an invalid/error response")
    return result


def social_weight(
    origin, destination, *, beta: float, omega: float, c0: float,
    modes=("WALK", "TRANSIT"), api_key: str | None = None,
    departure_time: datetime | None = None,
    driving_usd_per_km: float | None = None, parking_usd: float = 0,
    bicycle_fee_usd: float = 0, transit_fare_usd: float | None = None,
    driving_tolls_usd: float | None = None, timeout: float = 30,
    request_fn=None,
) -> SocialWeight:
    """Compute exp(-beta * max(c0, min_m(price_m + omega * hours_m))).

    Coordinates: (latitude, longitude), WGS84 degrees. All money is USD.
    omega is USD/hour; beta is 1/USD; c0 is USD. Modes must be available
    to the modeled person. Prices are per traveler, for a one-way trip.
    Explicit fare/toll values override API estimates. Missing prices raise.
    Empty route responses exclude a mode; API failures raise instead.
    Minimization is over routes returned by Google, not every possible route.
    Identical representative points use c0 without making an API request.
    request_fn can supply saved responses for fully offline recomputation.
    """
    start, end = _point(origin), _point(destination)
    for name, value in (("beta", beta), ("omega", omega), ("c0", c0),
                        ("parking_usd", parking_usd), ("bicycle_fee_usd", bicycle_fee_usd),
                        ("timeout", timeout)):
        _nonnegative(name, value)
    if beta == 0 or c0 == 0 or timeout == 0:
        raise ValueError("beta, c0 and timeout must be positive")
    for name, value in (("driving_usd_per_km", driving_usd_per_km),
                        ("transit_fare_usd", transit_fare_usd),
                        ("driving_tolls_usd", driving_tolls_usd)):
        if value is not None:
            _nonnegative(name, value)
    modes = tuple(dict.fromkeys(modes))
    if not modes or not set(modes) <= MODES:
        raise ValueError(f"Choose one or more modes from {sorted(MODES)}")
    departure = departure_time or datetime.now(timezone.utc)
    if departure.tzinfo is None or departure.utcoffset() is None:
        raise ValueError("departure_time must be timezone-aware")
    timestamp = departure.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if start == end:
        return SocialWeight(math.exp(-beta*c0), -beta*c0, c0, "LOCAL", (), (), timestamp)
    if "DRIVE" in modes and driving_usd_per_km is None:
        raise ValueError("Supply driving_usd_per_km when DRIVE is enabled")
    key = api_key or os.environ.get("GOOGLE_MAPS_API_KEY")
    if not key:
        raise ValueError("Set GOOGLE_MAPS_API_KEY with Google Routes API enabled")

    alternatives, unavailable = [], []
    for mode in modes:
        body = {"origin": start, "destination": end, "travelMode": mode,
                "computeAlternativeRoutes": True}
        mask = "routes.duration,routes.distanceMeters"
        if mode in {"WALK", "BICYCLE"}:
            mask += ",routes.legs.steps.navigationInstruction.maneuver"
        if mode in {"DRIVE", "TRANSIT"}:
            body["departureTime"] = timestamp
        if mode == "TRANSIT":
            mask += ",routes.travelAdvisory.transitFare"
        if mode == "DRIVE":
            body.update(routingPreference="TRAFFIC_AWARE", extraComputations=["TOLLS"])
            mask += ",routes.travelAdvisory.tollInfo"
        routes = (request_fn or _post)(body, key, mask, timeout).get("routes", [])
        if not routes:
            unavailable.append(mode)
        for route in routes:
            if mode in {"WALK", "BICYCLE"} and any(
                step.get("navigationInstruction", {}).get("maneuver", "").startswith("FERRY")
                for leg in route.get("legs", []) for step in leg.get("steps", [])
            ):
                raise MissingPriceError("Walking/cycling ferry route requires explicit fare and schedule modeling")
            duration = route.get("duration", "")
            if not isinstance(duration, str) or not duration.endswith("s"):
                raise RoutingError("Route lacks a valid duration")
            hours = float(duration[:-1]) / 3600
            # Protobuf may omit numeric zero fields.
            distance = float(route.get("distanceMeters", 0))
            _nonnegative("route duration", hours)
            _nonnegative("route distance", distance)
            advisory = route.get("travelAdvisory", {})
            if mode == "WALK":
                price = 0.0
            elif mode == "BICYCLE":
                price = bicycle_fee_usd
            elif mode == "TRANSIT":
                if transit_fare_usd is not None:
                    price = transit_fare_usd
                elif "transitFare" in advisory:
                    price = _money(advisory["transitFare"])
                else:
                    raise MissingPriceError("Transit fare unavailable; supply transit_fare_usd")
            else:
                if driving_tolls_usd is not None:
                    toll = driving_tolls_usd
                elif "tollInfo" not in advisory:
                    toll = 0.0
                else:
                    prices = advisory["tollInfo"].get("estimatedPrice", [])
                    if not prices:
                        raise MissingPriceError("Toll amount unavailable; supply driving_tolls_usd")
                    toll = sum(_money(p) for p in prices)
                price = distance / 1000 * driving_usd_per_km + parking_usd + toll
            generalized = price + omega * hours
            alternatives.append(ModeCost(mode, hours, distance, price, generalized))
    if not alternatives:
        # Convention: min(empty set) = +infinity, hence f = 0.
        return SocialWeight(0.0, -math.inf, math.inf, None, (), tuple(unavailable), timestamp)
    best = min(alternatives, key=lambda x: x.generalized_cost_usd)
    cost = max(c0, best.generalized_cost_usd)
    return SocialWeight(math.exp(-beta*cost), -beta*cost, cost, best.mode,
                        tuple(alternatives), tuple(unavailable), timestamp)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("origin", nargs=2, type=float, metavar=("LAT_I", "LON_I"))
    parser.add_argument("destination", nargs=2, type=float, metavar=("LAT_J", "LON_J"))
    for parameter in ("beta", "omega", "c0"):
        parser.add_argument(f"--{parameter}", required=True, type=float)
    parser.add_argument("--modes", nargs="+", choices=sorted(MODES), default=["WALK", "TRANSIT"])
    for parameter in ("driving-usd-per-km", "parking-usd", "bicycle-fee-usd",
                      "transit-fare-usd", "driving-tolls-usd"):
        parser.add_argument(f"--{parameter}", type=float)
    parser.add_argument("--departure-time", help="ISO 8601 timestamp with timezone")
    args = vars(parser.parse_args())
    args = {k: v for k, v in args.items() if v is not None}
    if "departure_time" in args:
        args["departure_time"] = datetime.fromisoformat(args["departure_time"].replace("Z", "+00:00"))
    result = asdict(social_weight(**args))
    # Strict JSON uses null for infinite costs/log weights of unreachable pairs.
    for name in ("cost_usd", "log_f"):
        if not math.isfinite(result[name]):
            result[name] = None
    print(json.dumps(result, indent=2, allow_nan=False))
