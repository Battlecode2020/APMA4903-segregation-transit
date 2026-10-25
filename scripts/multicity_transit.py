"""Local, schedule-derived expected travel times for tract-level city studies.

This is an approximation, not a timetable router. Each exact stop-sequence and
boarding-permission pattern has its own directed onboard states. Ride and dwell
times are medians of active trips; boarding costs half the mean headway observed
in the analysis window. Static GTFS files are read locally: this module contains
no network client and never calls a Routes API.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix, vstack
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class TransitParameters:
    reference_date: str = "2026-10-20"
    window_start: str = "07:00:00"
    window_end: str = "10:00:00"
    walking_speed_kmh: float = 4.8
    walking_detour: float = 1.3
    bus_access_km: float = 1.0
    rail_access_km: float = 3.0
    transfer_walk_km: float = 0.4
    entrance_seconds: float = 60.0
    transfer_buffer_seconds: float = 60.0
    dijkstra_batch_size: int = 16
    access_policy: str = "capped"
    pedestrian_barrier_policy: str = "reject"

    def validate(self):
        if not 0 <= gtfs_seconds(self.window_start) < gtfs_seconds(self.window_end) <= 86400:
            raise ValueError("Expected a nonempty analysis window within one service day")
        for field in ("walking_speed_kmh", "walking_detour", "bus_access_km", "rail_access_km"):
            if not np.isfinite(getattr(self, field)) or getattr(self, field) <= 0:
                raise ValueError(f"Invalid {field}")
        for field in ("transfer_walk_km", "entrance_seconds", "transfer_buffer_seconds"):
            if not np.isfinite(getattr(self, field)) or getattr(self, field) < 0:
                raise ValueError(f"Invalid {field}")
        if self.walking_detour < 1 or self.dijkstra_batch_size < 1:
            raise ValueError("Invalid detour or batch size")
        if self.access_policy not in ("capped", "uncapped"):
            raise ValueError("access_policy must be capped or uncapped")
        if self.pedestrian_barrier_policy not in ("reject", "flag_only"):
            raise ValueError("pedestrian_barrier_policy must be reject or flag_only")
        date.fromisoformat(self.reference_date)


def gtfs_seconds(value):
    """GTFS times can exceed 24:00; missing times return NaN for interpolation."""
    if pd.isna(value) or str(value).strip() == "":
        return np.nan
    parts = str(value).strip().split(":")
    if len(parts) != 3:
        raise ValueError(f"Invalid GTFS time: {value!r}")
    h, m, s = map(int, parts)
    if h < 0 or not 0 <= m < 60 or not 0 <= s < 60:
        raise ValueError(f"Invalid GTFS time: {value!r}")
    return h * 3600 + m * 60 + s


def _table(archive, name, required=False):
    candidates = [x for x in archive.namelist() if Path(x).name == name + ".txt"]
    if not candidates:
        if required:
            raise ValueError(f"Missing GTFS {name}.txt")
        return pd.DataFrame()
    if len(candidates) != 1:
        raise ValueError(f"Ambiguous GTFS {name}.txt")
    frame = pd.read_csv(archive.open(candidates[0]), dtype=str, keep_default_na=False)
    frame.columns = frame.columns.str.strip()
    for column in frame:
        frame[column] = frame[column].str.strip()
    return frame


def active_services(calendar, exceptions, day):
    stamp = day.strftime("%Y%m%d")
    result = set()
    if not calendar.empty:
        valid = (calendar.start_date <= stamp) & (calendar.end_date >= stamp)
        valid &= calendar[day.strftime("%A").lower()].eq("1")
        result.update(calendar.loc[valid, "service_id"])
    if not exceptions.empty:
        for row in exceptions.loc[exceptions.date.eq(stamp)].itertuples():
            if row.exception_type == "1":
                result.add(row.service_id)
            elif row.exception_type == "2":
                result.discard(row.service_id)
            else:
                raise ValueError("Invalid calendar exception type")
    return result


def _available_dates(calendar, exceptions):
    """Tuesday dates, with other weekdays as fallback for unusual feeds."""
    intervals = []
    if not calendar.empty:
        intervals = [(date.fromisoformat(str(a)), date.fromisoformat(str(b)))
                     for a, b in zip(calendar.start_date, calendar.end_date)]
    if not exceptions.empty:
        intervals += [(date.fromisoformat(str(d)), date.fromisoformat(str(d)))
                      for d in exceptions.loc[exceptions.exception_type.eq("1"), "date"]]
    result = set()
    for first, last in intervals:
        if (last - first).days > 3660:
            raise ValueError("Implausibly long GTFS calendar")
        result.update(first + timedelta(days=i) for i in range((last-first).days+1))
    return result


def choose_service_dates(feedpaths, reference_date="2026-10-20"):
    """Prefer one common valid Tuesday; log separate service snapshots otherwise."""
    reference = date.fromisoformat(reference_date)
    valid_dates = []
    for path in feedpaths:
        with ZipFile(path) as archive:
            calendar, exceptions = _table(archive, "calendar"), _table(archive, "calendar_dates")
            trips = _table(archive, "trips", True)
            service_ids = set(trips.service_id)
            calendar = calendar.loc[calendar.service_id.isin(service_ids)] if not calendar.empty else calendar
            exceptions = exceptions.loc[exceptions.service_id.isin(service_ids)] if not exceptions.empty else exceptions
        candidates = _available_dates(calendar, exceptions)
        # Cache by candidate rather than trusting an overall min/max feed range.
        valid_dates.append({d for d in candidates if active_services(calendar, exceptions, d)})
    if not valid_dates or any(not x for x in valid_dates):
        raise ValueError("A feed has no valid service dates")
    common = set.intersection(*valid_dates)
    common_tuesdays = {d for d in common if d.weekday() == 1}
    if common_tuesdays:
        selected = min(common_tuesdays, key=lambda d: (abs((d-reference).days), d))
        return [selected] * len(feedpaths)
    if common:
        weekdays = {d for d in common if d.weekday() < 5} or common
        selected = min(weekdays, key=lambda d: (abs((d-reference).days), d))
        return [selected] * len(feedpaths)
    result = []
    for candidates in valid_dates:
        preferred = {d for d in candidates if d.weekday() == 1}
        preferred = preferred or {d for d in candidates if d.weekday() < 5} or candidates
        result.append(min(preferred, key=lambda d: (abs((d-reference).days), d)))
    return result


def route_mode(route_type):
    value = int(route_type)
    if value in (0, 1, 2, 5, 7, 12) or 100 <= value < 200 or 400 <= value < 500 or 900 <= value < 1000 or value == 1400:
        return "rail"
    if value in (3, 11) or 200 <= value < 300 or 700 <= value < 900:
        return "bus"
    return None


def distance_km(left, right):
    """Broadcasted haversine distances; input columns are latitude, longitude."""
    a = np.radians(np.asarray(left, dtype=float))
    b = np.radians(np.asarray(right, dtype=float))
    dlat = a[:, None, 0] - b[None, :, 0]
    dlon = a[:, None, 1] - b[None, :, 1]
    h = np.sin(dlat / 2)**2 + np.cos(a[:, None, 0])*np.cos(b[None, :, 0])*np.sin(dlon / 2)**2
    return 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def _point_distances(left, right):
    a, b = np.radians(left), np.radians(right)
    h = np.sin((a[:, 0]-b[:, 0])/2)**2 + np.cos(a[:, 0])*np.cos(b[:, 0])*np.sin((a[:, 1]-b[:, 1])/2)**2
    return 6371.0088*2*np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def _xyz(coords):
    lat, lon = np.radians(coords).T
    return 6371.0088*np.column_stack((np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)))


def _walk_validity(left, right, land_geometry=None, water_geometry=None):
    """Optional WGS84 land coverage check; not a street-network path test.

    Input geometry must cover the relevant surroundings and exclude open water.
    City boundary or unmodified Census tract polygons are NOT suitable land masks.
    """
    if land_geometry is None and water_geometry is None:
        return np.ones(len(left), dtype=bool)
    import shapely
    lines = shapely.linestrings(np.stack((left[:, ::-1], right[:, ::-1]), axis=1))
    valid = np.ones(len(left), dtype=bool)
    if land_geometry is not None:
        valid &= np.asarray(shapely.covers(land_geometry, lines), dtype=bool)
    if water_geometry is not None:
        valid &= ~np.asarray(shapely.intersects(water_geometry, lines), dtype=bool)
    return valid


def _stop_times(archive, active_trip_ids):
    filename = next(x for x in archive.namelist() if Path(x).name == "stop_times.txt")
    wanted = {"trip_id", "arrival_time", "departure_time", "stop_id", "stop_sequence",
              "pickup_type", "drop_off_type", "shape_dist_traveled"}
    chunks = []
    for chunk in pd.read_csv(archive.open(filename), dtype=str, keep_default_na=False,
                             usecols=lambda x: x.strip() in wanted, chunksize=200000):
        chunk.columns = chunk.columns.str.strip()
        for column in chunk:
            chunk[column] = chunk[column].str.strip()
        subset = chunk.loc[chunk.trip_id.isin(active_trip_ids)]
        if len(subset):
            chunks.append(subset)
    if not chunks:
        return pd.DataFrame()
    times = pd.concat(chunks, ignore_index=True)
    for field in ("pickup_type", "drop_off_type"):
        if field not in times:
            times[field] = "0"
        times[field] = times[field].replace("", "0")
        if not times[field].isin(["0", "1", "2", "3"]).all():
            raise ValueError(f"Unsupported {field}")
    times["sequence"] = pd.to_numeric(times.stop_sequence)
    for field in ("arrival", "departure"):
        times[field] = times[field+"_time"].map(gtfs_seconds)
    return times.sort_values(["trip_id", "sequence"])


def load_patterns(path, service_date, parameters):
    """Read only scheduled bus/rail trips active on one local service date."""
    start, end = gtfs_seconds(parameters.window_start), gtfs_seconds(parameters.window_end)
    with ZipFile(path) as archive:
        calendar, exceptions = _table(archive, "calendar"), _table(archive, "calendar_dates")
        services = active_services(calendar, exceptions, service_date)
        routes = _table(archive, "routes", True)
        routes["mode"] = routes.route_type.map(route_mode)
        route_modes = dict(zip(routes.route_id, routes["mode"]))
        trips = _table(archive, "trips", True)
        trips = trips.loc[trips.service_id.isin(services) & trips.route_id.isin(routes.loc[routes["mode"].notna(), "route_id"])]
        times = _stop_times(archive, set(trips.trip_id))
        stops = _table(archive, "stops", True)
        frequencies = _table(archive, "frequencies")
        transfers = _table(archive, "transfers")
        agency = _table(archive, "agency")
        feed_info = _table(archive, "feed_info")
    if times.empty:
        raise ValueError(f"No active bus/rail stop times in {path}")
    trip_routes = dict(zip(trips.trip_id, trips.route_id))
    freq_by_trip = {}
    if not frequencies.empty:
        for trip_id, frame in frequencies.groupby("trip_id", sort=False):
            intervals = [(gtfs_seconds(r.start_time), gtfs_seconds(r.end_time), float(r.headway_secs)) for r in frame.itertuples()]
            if any(not np.isfinite(a+b+h) or b <= a or h <= 0 for a, b, h in intervals):
                raise ValueError("Invalid frequency interval")
            ordered = sorted(intervals)
            if any(a[1] > b[0] for a, b in zip(ordered, ordered[1:])):
                raise ValueError("Overlapping frequency intervals for one trip")
            freq_by_trip[trip_id] = intervals
    patterns = {}
    counts = defaultdict(int)
    for trip_id, frame in times.groupby("trip_id", sort=False):
        if frame.sequence.duplicated().any():
            raise ValueError("Duplicate stop sequence within trip")
        arrivals = frame.arrival.to_numpy(dtype=float)
        departures = frame.departure.to_numpy(dtype=float)
        # GTFS allows missing times at intermediate stops. Interpolate monotonically
        # by shape distance where usable, otherwise by stop order, and record this.
        arrivals = np.where(np.isnan(arrivals), departures, arrivals)
        departures = np.where(np.isnan(departures), arrivals, departures)
        if np.isnan(arrivals[[0, -1]]).any() or np.isnan(departures[[0, -1]]).any():
            counts["trips_skipped_missing_endpoint_times"] += 1
            continue
        if np.isnan(arrivals).any() or np.isnan(departures).any():
            counts["trips_with_interpolated_stop_times"] += 1
            x = np.arange(len(frame), dtype=float)
            if "shape_dist_traveled" in frame:
                shape_x = pd.to_numeric(frame.shape_dist_traveled, errors="coerce").to_numpy()
                if np.isfinite(shape_x).all() and (np.diff(shape_x) > 0).all():
                    x = shape_x
            arrivals = np.interp(x, x[np.isfinite(arrivals)], arrivals[np.isfinite(arrivals)])
            departures = np.interp(x, x[np.isfinite(departures)], departures[np.isfinite(departures)])
        if (departures < arrivals).any() or (arrivals[1:] < departures[:-1]).any():
            counts["trips_skipped_nonmonotone_times"] += 1
            continue
        if len(frame) < 2:
            continue
        freq = freq_by_trip.get(trip_id)
        if freq is None and not (departures[0] < end and arrivals[-1] >= start):
            continue
        if freq is not None and not any(a < end and b+arrivals[-1]-departures[0] > start for a, b, _ in freq):
            continue
        route_id = trip_routes[trip_id]
        stop_ids = tuple(frame.stop_id)
        pickup = tuple(frame.pickup_type.eq("0"))
        dropoff = tuple(frame.drop_off_type.eq("0"))
        key = (route_id, stop_ids, pickup, dropoff)
        if key not in patterns:
            patterns[key] = {"route_id": route_id, "mode": route_modes[route_id], "stop_ids": stop_ids,
                             "pickup": pickup, "dropoff": dropoff, "rides": [], "dwells": [],
                             "departures_in_window": np.zeros(len(frame)), "trip_count": 0}
        p = patterns[key]
        p["rides"].append(arrivals[1:]-departures[:-1])
        p["dwells"].append(departures-arrivals)
        p["trip_count"] += 1
        if freq is None:
            p["departures_in_window"] += (departures >= start) & (departures < end)
            counts["scheduled_trip_samples"] += 1
        else:
            offsets = departures-departures[0]
            for a, b, headway in freq:
                p["departures_in_window"] += np.maximum(0, np.minimum(end, b+offsets)-np.maximum(start, a+offsets))/headway
            counts["frequency_trip_templates"] += 1
    result = []
    for p in patterns.values():
        if p["departures_in_window"].max() <= 0:
            continue
        p["ride_seconds"] = np.median(p.pop("rides"), axis=0)
        p["dwell_seconds"] = np.median(p.pop("dwells"), axis=0)
        count = p.pop("departures_in_window")
        p["boarding_wait_seconds"] = np.divide(end-start, 2*count, out=np.full(len(count), np.inf), where=count > 0)
        result.append(p)
    if not result:
        raise ValueError(f"No bus/rail service within analysis window in {path}")
    stops = stops.copy()
    stops["stop_lat"] = pd.to_numeric(stops.stop_lat, errors="coerce")
    stops["stop_lon"] = pd.to_numeric(stops.stop_lon, errors="coerce")
    if "parent_station" not in stops:
        stops["parent_station"] = ""
    parentcoords = stops.set_index("stop_id")[["stop_lat", "stop_lon"]]
    for field in ("stop_lat", "stop_lon"):
        stops[field] = stops[field].fillna(stops.parent_station.map(parentcoords[field]))
    metadata = {"path": str(path), "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                "service_date": service_date.isoformat(), "patterns": len(result),
                "route_modes": sorted({p["mode"] for p in result}),
                "used_routes_by_mode": {mode: len({p["route_id"] for p in result if p["mode"] == mode}) for mode in ("bus", "rail")},
                "published_feed_info": feed_info.to_dict("records"),
                "agency_timezones": sorted(set(agency.agency_timezone)) if "agency_timezone" in agency else [],
                "diagnostics": dict(counts)}
    return stops, result, transfers, metadata


def _uncapped_transit(edges, graphsize, nodecoords, stopcoords, boarding, alighting,
                      legacy_radius_km, parameters, land_geometry, water_geometry):
    """Unlimited access/egress using temporary batched origins, never n×S dicts.

    Origins connect only to boarding states. Destinations are evaluated only
    from genuine alighting states reached after a GTFS ride. Existing transfer
    and onboard graph edges are untouched. All access/egress time is charged.
    """
    n, ns = len(nodecoords), len(stopcoords)
    transit = np.full((n, n), np.inf)
    details = {
        "transit_access_seconds": np.full((n, n), np.inf),
        "transit_egress_seconds": np.full((n, n), np.inf),
        "transit_first_boarding_stop_index": np.full((n, n), -1, dtype=np.int32),
        "transit_last_alighting_stop_index": np.full((n, n), -1, dtype=np.int32),
        "transit_access_barrier_flag": np.zeros((n, n), dtype=bool),
        "transit_egress_barrier_flag": np.zeros((n, n), dtype=bool),
        "transit_access_exceeds_legacy_radius": np.zeros((n, n), dtype=bool),
        "transit_egress_exceeds_legacy_radius": np.zeros((n, n), dtype=bool),
    }
    diagnostics = {"routing_algorithm": "batched temporary origins plus chunked min-plus egress",
                   "base_graph_nodes": graphsize, "base_graph_edges": len(edges),
                   "maximum_temporary_access_edges": 0, "maximum_temporary_origins": 0}
    board_stops = np.array([s for s in sorted(boarding) if boarding[s]], dtype=int)
    arrival_stops = np.array([s for s in sorted(alighting) if alighting[s]], dtype=int)
    if not len(board_stops) or not len(arrival_stops):
        return transit, np.zeros(n, dtype=int), details, diagnostics
    print(f"  Computing uncapped walking access/egress for {n:,} × {ns:,} tract-stop pairs", flush=True)
    walk_to_stop = np.empty((n, ns), dtype=float)
    checked = land_geometry is not None or water_geometry is not None
    reject = parameters.pedestrian_barrier_policy == "reject" and checked
    for i, coord in enumerate(nodecoords):
        walk_to_stop[i] = distance_km(coord[None, :], stopcoords)[0]*parameters.walking_detour*3600/parameters.walking_speed_kmh+parameters.entrance_seconds
        if reject:
            valid = _walk_validity(np.repeat(coord[None, :], ns, axis=0), stopcoords, land_geometry, water_geometry)
            walk_to_stop[i, ~valid] = np.inf
        if i+1 == n or (i+1) % 100 == 0:
            print(f"    Access/egress screened for {i+1:,}/{n:,} origins", flush=True)
    access_count = np.isfinite(walk_to_stop[:, board_stops]).sum(axis=1)
    boarding_states, boarding_stop_ids, waits = [], [], []
    for s in board_stops:
        for state, wait in boarding[s]:
            boarding_states.append(state); boarding_stop_ids.append(s); waits.append(wait)
    order = np.argsort(boarding_states)
    boarding_states = np.asarray(boarding_states, dtype=int)[order]
    boarding_stop_ids = np.asarray(boarding_stop_ids, dtype=int)[order]
    waits = np.asarray(waits, dtype=float)[order]
    stop_of_boarding_state = np.full(graphsize, -1, dtype=int)
    stop_of_boarding_state[boarding_states] = boarding_stop_ids
    arrival_states = np.concatenate([np.asarray(alighting[s], dtype=int) for s in arrival_stops])
    starts = np.cumsum([0]+[len(alighting[s]) for s in arrival_stops[:-1]])
    if edges:
        row, col = np.array(list(edges), dtype=int).T
        base_graph = coo_matrix((np.fromiter(edges.values(), dtype=float), (row, col)), shape=(graphsize, graphsize)).tocsr()
    else:
        base_graph = csr_matrix((graphsize, graphsize))
    print(f"  Uncapped transit paths: {graphsize:,} base nodes, {len(edges):,} base edges", flush=True)
    for first in range(0, n, parameters.dijkstra_batch_size):
        last = min(n, first+parameters.dijkstra_batch_size)
        batchsize = last-first
        seed_costs = walk_to_stop[first:last, boarding_stop_ids]+waits[None, :]
        finite_seeds = np.isfinite(seed_costs)
        rowptr = np.r_[0, np.cumsum(finite_seeds.sum(axis=1))]
        access_graph = csr_matrix((np.maximum(seed_costs[finite_seeds], 1e-8),
                                  np.broadcast_to(boarding_states, seed_costs.shape)[finite_seeds], rowptr),
                                 shape=(batchsize, graphsize+batchsize))
        padded_base = csr_matrix((base_graph.data, base_graph.indices, base_graph.indptr),
                                shape=(graphsize, graphsize+batchsize))
        graph = vstack([padded_base, access_graph], format="csr")
        diagnostics["maximum_temporary_access_edges"] = max(diagnostics["maximum_temporary_access_edges"], int(access_graph.nnz))
        diagnostics["maximum_temporary_origins"] = max(diagnostics["maximum_temporary_origins"], batchsize)
        distances, predecessors = dijkstra(graph, directed=True,
            indices=np.arange(graphsize, graphsize+batchsize), return_predecessors=True)
        arrivals = np.minimum.reduceat(distances[:, arrival_states], starts, axis=1)
        selected_egress = np.full((batchsize, n), -1, dtype=int)
        # Bound temporary B×destination-chunk×stops arrays to roughly 32 MiB.
        destination_chunk = max(1, min(32, 4_000_000//max(1, batchsize*len(arrival_stops))))
        for j in range(0, n, destination_chunk):
            end = min(n, j+destination_chunk)
            candidate = arrivals[:, None, :] + walk_to_stop[j:end, arrival_stops][None, :, :]
            best = candidate.argmin(axis=2)
            values = np.take_along_axis(candidate, best[:, :, None], axis=2)[:, :, 0]
            transit[first:last, j:end] = values
            selected_egress[:, j:end] = np.where(np.isfinite(values), arrival_stops[best], -1)
        # Recover first boarding stops only for selected endpoints, memoizing
        # the predecessor forest. This supplies auditable access/egress times.
        for b in range(batchsize):
            i = first+b
            selected_egress[b, i] = -1
            roots = np.full(graphsize, -2, dtype=int)
            def first_boarding_state(state):
                trail = []
                while roots[state] == -2:
                    parent = int(predecessors[b, state])
                    if parent == graphsize+b:
                        roots[state] = state
                        break
                    if parent < 0 or parent >= graphsize:
                        raise ValueError("Transit path did not start with a real boarding edge")
                    trail.append(state)
                    state = parent
                root = int(roots[state])
                if trail:
                    roots[np.asarray(trail, dtype=int)] = root
                return root
            for s in np.unique(selected_egress[b][selected_egress[b] >= 0]):
                ends = np.asarray(alighting[int(s)], dtype=int)
                end_state = int(ends[np.argmin(distances[b, ends])])
                start_state = first_boarding_state(end_state)
                board_stop = int(stop_of_boarding_state[start_state])
                if board_stop < 0:
                    raise ValueError("Recovered transit source is not a boarding state")
                destinations = np.flatnonzero(selected_egress[b] == s)
                details["transit_first_boarding_stop_index"][i, destinations] = board_stop
                details["transit_last_alighting_stop_index"][i, destinations] = s
                details["transit_access_seconds"][i, destinations] = walk_to_stop[i, board_stop]
                details["transit_egress_seconds"][i, destinations] = walk_to_stop[destinations, s]
                access_km = (walk_to_stop[i, board_stop]-parameters.entrance_seconds)*parameters.walking_speed_kmh/3600
                egress_km = (walk_to_stop[destinations, s]-parameters.entrance_seconds)*parameters.walking_speed_kmh/3600
                details["transit_access_exceeds_legacy_radius"][i, destinations] = access_km > legacy_radius_km[board_stop]+1e-12
                details["transit_egress_exceeds_legacy_radius"][i, destinations] = egress_km > legacy_radius_km[s]+1e-12
                if checked and not reject:
                    access_valid = _walk_validity(nodecoords[[i]], stopcoords[[board_stop]], land_geometry, water_geometry)[0]
                    egress_valid = _walk_validity(nodecoords[destinations], np.repeat(stopcoords[[s]], len(destinations), axis=0), land_geometry, water_geometry)
                    details["transit_access_barrier_flag"][i, destinations] = ~access_valid
                    details["transit_egress_barrier_flag"][i, destinations] = ~egress_valid
        if last == n or first//100 != last//100:
            print(f"    {last:,}/{n:,} uncapped tract origins", flush=True)
    np.fill_diagonal(transit, np.inf)
    finite = np.isfinite(transit)
    charged_walk = details["transit_access_seconds"][finite]+details["transit_egress_seconds"][finite]
    if (transit[finite]+1e-7 < charged_walk).any():
        raise ValueError("Reported transit time does not include all access/egress time")
    diagnostics["finite_journeys_with_access_beyond_legacy_radius"] = int((finite & details["transit_access_exceeds_legacy_radius"]).sum())
    diagnostics["finite_journeys_with_egress_beyond_legacy_radius"] = int((finite & details["transit_egress_exceeds_legacy_radius"]).sum())
    return transit, access_count, details, diagnostics


def compute_city(nodes, feedpaths, outputdir, *, parameters=None, land_geometry=None, water_geometry=None, force=False):
    """Write modes.npz and manifest.json, returning the latter dictionary.

    ``nodes`` is a DataFrame or CSV with tract_geoid, latitude, longitude. If
    supplied, ``land_geometry`` is a WGS84 Shapely geometry excluding water.
    ``water_geometry`` can instead identify water barriers. Both are WGS84.
    ``walk_seconds`` always holds the complete, explicitly approximate distance
    proxy; ``walk_barrier_checked_seconds`` rejects chords failing available
    geometry checks. Default capped access retains the original behavior.
    ``access_policy='uncapped'`` considers every stop for walking access/egress
    with full walking time charged; ``pedestrian_barrier_policy='reject'`` still
    excludes chords failing supplied geometry. ``flag_only`` retains those
    access/egress proxies and flags selected legs. Inter-stop transfer walks
    always retain the original radius and geometry checks. Transit NEVER has a
    fabricated fallback, and requires a real ride. Its diagonal is infinite.
    """
    parameters = parameters or TransitParameters()
    parameters.validate()
    # An empty land mask would reject every access/egress leg without an error,
    # producing plausible-looking all-walking scores from a broken input filter.
    # Validate before cache lookup so an earlier bad result cannot be reused.
    for label, geometry in (("land", land_geometry), ("water", water_geometry)):
        if geometry is None:
            continue
        if not hasattr(geometry, "is_valid") or not hasattr(geometry, "is_empty"):
            raise ValueError(f"Expected a Shapely {label} geometry")
        if not geometry.is_valid:
            raise ValueError(f"Supplied {label} geometry is invalid; repair or replace the input before routing")
        if label == "land" and geometry.is_empty:
            raise ValueError("Supplied land geometry is empty; refusing to silently exclude every walking/transit access leg")
    # An explicitly empty, valid water mask can legitimately represent no mapped
    # water barriers. It does not claim that streets or other barriers were checked.
    nodes = pd.read_csv(nodes, dtype={"tract_geoid": str}) if isinstance(nodes, (str, Path)) else nodes.copy()
    nodes = nodes.reset_index(drop=True)
    if "matrix_index" in nodes and not np.array_equal(nodes.matrix_index.to_numpy(), np.arange(len(nodes))):
        raise ValueError("nodes must be in matrix_index order")
    if nodes.tract_geoid.duplicated().any():
        raise ValueError("Duplicate tract identifiers")
    nodecoords = nodes[["latitude", "longitude"]].to_numpy(dtype=float)
    if not np.isfinite(nodecoords).all() or (np.abs(nodecoords[:, 0]) > 90).any() or (np.abs(nodecoords[:, 1]) > 180).any():
        raise ValueError("Invalid tract coordinates")
    feedpaths = [Path(p) for p in feedpaths]
    outputdir = Path(outputdir)
    outputdir.mkdir(parents=True, exist_ok=True)
    input_fingerprint = hashlib.sha256(nodes[["tract_geoid", "latitude", "longitude"]].to_csv(index=False).encode()).hexdigest()
    fingerprint = {"nodes_sha256": input_fingerprint, "parameters": asdict(parameters),
                   "feeds": [hashlib.sha256(p.read_bytes()).hexdigest() for p in feedpaths],
                   "land_sha256": hashlib.sha256(land_geometry.wkb).hexdigest() if land_geometry is not None else None,
                   "water_sha256": hashlib.sha256(water_geometry.wkb).hexdigest() if water_geometry is not None else None,
                   "model_version": 4}
    manifest_path = outputdir/"manifest.json"
    if not force and manifest_path.exists() and (outputdir/"modes.npz").exists():
        existing = json.loads(manifest_path.read_text())
        if existing.get("input_fingerprint") == fingerprint:
            print(f"  Using cached travel matrices for {len(nodes)} tracts", flush=True)
            return existing
    # Prepared geometry caches a spatial index without changing its coordinates
    # or any predicate results. Repeated line tests otherwise rescan thousands
    # of water polygon boundaries for every tract/stop pair.
    if land_geometry is not None or water_geometry is not None:
        import shapely
        for geometry in (land_geometry, water_geometry):
            if geometry is not None:
                shapely.prepare(geometry)
    print(f"  Selecting service dates for {len(feedpaths)} feeds", flush=True)
    service_dates = choose_service_dates(feedpaths, parameters.reference_date) if feedpaths else []
    all_stops, all_patterns, feed_metadata, transfer_tables = [], [], [], []
    for feed_index, (path, service_date) in enumerate(zip(feedpaths, service_dates)):
        print(f"  Loading {path.name} for {service_date.isoformat()}", flush=True)
        stops, patterns, transfers, metadata = load_patterns(path, service_date, parameters)
        print(f"    {len(patterns):,} service patterns", flush=True)
        prefix = f"{feed_index}:"
        stops["stop_id"] = prefix+stops.stop_id
        stops["parent_station"] = stops.parent_station.map(lambda x: prefix+x if x else "")
        for p in patterns:
            p["stop_ids"] = tuple(prefix+x for x in p["stop_ids"])
            p["route_id"] = prefix+p["route_id"]
        if not transfers.empty:
            for field in ("from_stop_id", "to_stop_id"):
                transfers[field] = prefix+transfers[field]
            for field in ("from_route_id", "to_route_id", "from_trip_id", "to_trip_id"):
                if field in transfers:
                    transfers[field] = transfers[field].map(lambda x: prefix+x if x else "")
            transfer_tables.append(transfers)
        all_stops.append(stops)
        all_patterns.extend(patterns)
        feed_metadata.append(metadata)
    used_stops = {s for p in all_patterns for s in p["stop_ids"]}
    stops = pd.concat(all_stops, ignore_index=True) if all_stops else pd.DataFrame(columns=["stop_id", "stop_lat", "stop_lon", "parent_station"])
    stops = stops.loc[stops.stop_id.isin(used_stops)].reset_index(drop=True)
    stopcoords = stops[["stop_lat", "stop_lon"]].to_numpy(dtype=float)
    if not np.isfinite(stopcoords).all():
        raise ValueError("Active service has stops with missing coordinates")
    stopindex = dict(zip(stops.stop_id, range(len(stops))))
    n, ns = len(nodes), len(stops)
    print(f"  Building graph: {len(all_patterns):,} patterns, {ns:,} physical stops", flush=True)
    edges = {}
    def edge(u, v, cost):
        if np.isfinite(cost):
            edges[(int(u), int(v))] = min(edges.get((int(u), int(v)), np.inf), max(float(cost), 1e-8))
    boarding = defaultdict(list)
    alighting = defaultdict(list)
    arrival_routes = {}
    mode_by_stop = defaultdict(set)
    next_index = ns
    for p in all_patterns:
        ids = [stopindex[s] for s in p["stop_ids"]]
        arrivals = np.arange(next_index, next_index+len(ids))
        departures = arrivals+len(ids)
        next_index += 2*len(ids)
        for k, s in enumerate(ids):
            mode_by_stop[s].add(p["mode"])
            if p["pickup"][k] and np.isfinite(p["boarding_wait_seconds"][k]) and k < len(ids)-1:
                wait = float(p["boarding_wait_seconds"][k])
                edge(s, departures[k], wait)
                boarding[s].append((departures[k], wait))
            if p["dropoff"][k] and k > 0:
                edge(arrivals[k], s, parameters.transfer_buffer_seconds)
                alighting[s].append(arrivals[k])
                arrival_routes[int(arrivals[k])] = p["route_id"]
            if 0 < k < len(ids)-1:
                edge(arrivals[k], departures[k], p["dwell_seconds"][k])
            if k < len(ids)-1:
                edge(departures[k], arrivals[k+1], p["ride_seconds"][k])
    # Pedestrian transfers between nearby actual stops, including separate feeds.
    pairs = np.asarray(list(cKDTree(_xyz(stopcoords)).query_pairs(parameters.transfer_walk_km/parameters.walking_detour)), dtype=int).reshape(-1, 2) if ns else np.empty((0, 2), dtype=int)
    transfer_pairs = 0
    if len(pairs):
        distances = _point_distances(stopcoords[pairs[:, 0]], stopcoords[pairs[:, 1]])
        valid = _walk_validity(stopcoords[pairs[:, 0]], stopcoords[pairs[:, 1]], land_geometry, water_geometry)
        for (a, b), distance, allowed in zip(pairs, distances, valid):
            if allowed:
                cost = distance*parameters.walking_detour*3600/parameters.walking_speed_kmh
                edge(a, b, cost)
                edge(b, a, cost)
                transfer_pairs += 1
    # Apply restrictive transfer rules BEFORE entering the physical-stop layer,
    # so walking via a third stop cannot bypass them. Their scope is deliberately
    # conservative: a minimum applies to all onward transfers from affected
    # arrivals, and a prohibition disables all onward transfers from those
    # arrivals. Direct destination egress and staying onboard remain possible.
    parent_children = stops.groupby("parent_station").stop_id.apply(list).to_dict() if ns else {}
    transfer_rules_applied = 0
    transfer_rule_diagnostics = defaultdict(int)
    for table in transfer_tables:
        for r in table.fillna("").to_dict("records"):
            kind = r.get("transfer_type", "0") or "0"
            if kind in ("4", "5"):
                transfer_rule_diagnostics["in_seat_transfer_rules_not_modeled"] += 1
                continue
            if kind not in ("0", "1", "2", "3"):
                raise ValueError("Unsupported GTFS transfer type")
            if kind == "1":
                transfer_rule_diagnostics["timed_transfer_guarantees_not_modeled"] += 1
            sources = parent_children.get(r["from_stop_id"], [r["from_stop_id"]])
            targets = parent_children.get(r["to_stop_id"], [r["to_stop_id"]])
            for source in sources:
                if source not in stopindex:
                    continue
                a = stopindex[source]
                affected_arrivals = [arrival for arrival in alighting[a]
                                     if not r.get("from_route_id", "") or arrival_routes[int(arrival)] == r["from_route_id"]]
                if kind == "3":
                    for arrival in affected_arrivals:
                        edges.pop((int(arrival), a), None)
                    transfer_rule_diagnostics["prohibitions_conservatively_applied_to_all_onward_transfers"] += 1
                elif kind == "2":
                    minimum = float(r.get("min_transfer_time", "0") or "0")
                    if not np.isfinite(minimum) or minimum < 0:
                        raise ValueError("Invalid minimum transfer time")
                    for arrival in affected_arrivals:
                        if (int(arrival), a) in edges:
                            edges[(int(arrival), a)] = max(edges[(int(arrival), a)], minimum)
                    transfer_rule_diagnostics["minimums_conservatively_applied_to_all_onward_transfers"] += 1
                for target in targets:
                    if target not in stopindex:
                        continue
                    b = stopindex[target]
                    if a != b and kind != "3":
                        if _walk_validity(stopcoords[[a]], stopcoords[[b]], land_geometry, water_geometry)[0]:
                            duration = _point_distances(stopcoords[[a]], stopcoords[[b]])[0]*parameters.walking_detour*3600/parameters.walking_speed_kmh
                            edge(a, b, duration)
                    transfer_rules_applied += 1
    geodesic = distance_km(nodecoords, nodecoords)
    walk_proxy = geodesic*parameters.walking_detour*3600/parameters.walking_speed_kmh
    walk_supported = np.zeros((n, n), dtype=bool)
    barrier_check_available = land_geometry is not None or water_geometry is not None
    print(f"  Screening {n*n:,} walking pairs against available water/land geometry", flush=True)
    if barrier_check_available:
        for i in range(n):
            walk_supported[i] = _walk_validity(np.repeat(nodecoords[[i]], n, axis=0), nodecoords, land_geometry, water_geometry)
        np.fill_diagonal(walk_supported, True)
    walking = np.where(walk_supported, walk_proxy, np.inf) if barrier_check_available else walk_proxy.copy()
    np.fill_diagonal(walking, 0)
    origin_offset = next_index
    transit_details, routing_diagnostics = {}, {}
    if parameters.access_policy == "uncapped":
        legacy_radius = np.array([parameters.rail_access_km if "rail" in mode_by_stop[s] else parameters.bus_access_km for s in range(ns)])
        transit, access_count, transit_details, routing_diagnostics = _uncapped_transit(
            edges, origin_offset, nodecoords, stopcoords, boarding, alighting,
            legacy_radius, parameters, land_geometry, water_geometry)
        graphsize = origin_offset+min(n, parameters.dijkstra_batch_size)
    else:
        access_by_stop = defaultdict(list)
        access_count = np.zeros(n, dtype=int)
        print("  Connecting tract access/egress walking legs", flush=True)
        if ns:
            tree = cKDTree(_xyz(stopcoords))
            for i, coord in enumerate(nodecoords):
                candidates = np.asarray(tree.query_ball_point(_xyz(nodecoords[[i]])[0], max(parameters.bus_access_km, parameters.rail_access_km)/parameters.walking_detour), dtype=int)
                if not len(candidates): continue
                distances = _point_distances(np.repeat(coord[None, :], len(candidates), axis=0), stopcoords[candidates])
                limits = np.array([parameters.rail_access_km if "rail" in mode_by_stop[s] else parameters.bus_access_km for s in candidates])
                valid = distances*parameters.walking_detour <= limits
                if parameters.pedestrian_barrier_policy == "reject":
                    valid &= _walk_validity(np.repeat(coord[None, :], len(candidates), axis=0), stopcoords[candidates], land_geometry, water_geometry)
                for s, distance in zip(candidates[valid], distances[valid]):
                    duration = distance*parameters.walking_detour*3600/parameters.walking_speed_kmh+parameters.entrance_seconds
                    access_by_stop[int(s)].append((i, duration))
                    for departure, wait in boarding[s]: edge(origin_offset+i, departure, duration+wait)
                access_count[i] = sum(bool(boarding[s]) for s in candidates[valid])
        graphsize = origin_offset+n
        if edges:
            row, col = np.array(list(edges), dtype=int).T
            graph = coo_matrix((np.fromiter(edges.values(), dtype=float), (row, col)), shape=(graphsize, graphsize)).tocsr()
        else:
            graph = coo_matrix((graphsize, graphsize)).tocsr()
        transit = np.full((n, n), np.inf)
        print(f"  Computing transit paths: {graphsize:,} graph nodes, {len(edges):,} edges", flush=True)
        for first in range(0, n, parameters.dijkstra_batch_size):
            last = min(n, first+parameters.dijkstra_batch_size)
            distances = dijkstra(graph, directed=True, indices=np.arange(origin_offset+first, origin_offset+last))
            for s, destinations in access_by_stop.items():
                if not alighting[s]: continue
                arrival = distances[:, alighting[s]].min(axis=1)
                for j, egress in destinations:
                    transit[first:last, j] = np.minimum(transit[first:last, j], arrival+egress)
            if last == n or first//100 != last//100:
                print(f"    {last:,}/{n:,} tract origins", flush=True)
    np.fill_diagonal(transit, np.inf)
    np.savez_compressed(outputdir/"modes.npz", tract_geoid=nodes.tract_geoid.astype(str).to_numpy(dtype=str),
                        matrix_index=np.arange(n), walk_seconds=walk_proxy, walk_proxy_seconds=walk_proxy,
                        walk_barrier_checked_seconds=walking,
                        walk_land_supported=walk_supported, walk_land_check_available=np.array(land_geometry is not None),
                        walk_barrier_check_available=np.array(barrier_check_available),
                        walk_water_crossing=(~walk_supported if water_geometry is not None else np.zeros((n, n), dtype=bool)),
                        walk_water_check_available=np.array(water_geometry is not None),
                        transit_seconds=transit, great_circle_distance_km=geodesic, access_stop_count=access_count,
                        transit_stop_ids=stops.stop_id.astype(str).to_numpy(dtype=str), **transit_details)
    nonlocal_mask = ~np.eye(n, dtype=bool)
    manifest = {"model": "GTFS pattern graph with mean-headway waiting", "input_fingerprint": fingerprint,
                "parameters": asdict(parameters), "feeds": feed_metadata, "nodes": n,
                "graph_nodes": graphsize, "graph_edges": len(edges), "service_patterns": len(all_patterns),
                "served_physical_stops": ns, "pedestrian_transfer_pairs": transfer_pairs,
                "explicit_transfer_rules_applied": transfer_rules_applied,
                "transfer_rule_diagnostics": dict(transfer_rule_diagnostics),
                "access_policy": parameters.access_policy,
                "pedestrian_barrier_policy": parameters.pedestrian_barrier_policy,
                "walking_access_interpretation": "No maximum walking distance for initial access or final egress; every accepted walking second is charged." if parameters.access_policy == "uncapped" else "Legacy initial-access and final-egress radii apply.",
                "transfer_walking_interpretation": f"Inter-stop transfer walks retain their {parameters.transfer_walk_km:g} km approximate-walking radius and supplied geometry checks; uncapped access does not remove this transfer restriction.",
                "routing_diagnostics": routing_diagnostics,
                "mode_array_definitions": {
                    "transit_seconds": "Minimum modeled walk-access + boarding wait + actual GTFS ride(s), dwell/transfers + walk-egress. At least one ride required; diagonal and absent candidates are infinity.",
                    "transit_access_seconds": "For uncapped output: selected initial walking time plus entrance_seconds, fully included in transit_seconds.",
                    "transit_egress_seconds": "For uncapped output: selected final walking time plus entrance_seconds exit buffer, fully included in transit_seconds.",
                    "transit_first_boarding_stop_index": "For uncapped output: zero-based index into transit_stop_ids; -1 when no transit candidate or on the diagonal.",
                    "transit_last_alighting_stop_index": "For uncapped output: zero-based index into transit_stop_ids; -1 when no transit candidate or on the diagonal.",
                    "transit_access_barrier_flag": "For uncapped flag_only output: selected access chord fails supplied land/water screen; false under reject policy because such legs are excluded. False is not validation against streets or barriers outside supplied geometry.",
                    "transit_egress_barrier_flag": "Same as transit_access_barrier_flag, for selected egress chord.",
                    "transit_access_exceeds_legacy_radius": "For uncapped output: selected initial walking distance exceeds old mode-specific bus_access_km/rail_access_km radius. Exit/entrance buffer is excluded from this distance comparison.",
                    "transit_egress_exceeds_legacy_radius": "Same as transit_access_exceeds_legacy_radius, for final walking leg."},
                "common_service_date": len(set(service_dates)) <= 1,
                "nonlocal_pairs": int(nonlocal_mask.sum()),
                "transit_reachable_nonlocal_pairs": int((np.isfinite(transit) & nonlocal_mask).sum()),
                "tracts_without_access_stops": int((access_count == 0).sum()),
                "walk_land_check_available": land_geometry is not None,
                "walk_water_check_available": water_geometry is not None,
                "walk_barrier_check_available": barrier_check_available,
                "barrier_clear_nonlocal_walk_pairs": int((walk_supported & nonlocal_mask).sum()) if barrier_check_available else None,
                "limitations": [
                    "Expected-time graph, not exact departure-time or real-time routing; median rides and half mean headway are approximations.",
                    "Exact stop-sequence patterns are separate: splitting service patterns can overestimate waiting when several patterns serve the same journey.",
                    "Walking is great-circle distance times a detour factor, not a street/bridge/pedestrian-access route.",
                    "walk_seconds is the complete distance approximation, including potentially impossible pedestrian water crossings; use walk_barrier_checked_seconds for conservative barrier filtering.",
                    "Without supplied land/water geometry, pedestrian legs have not been checked for water barriers. Supplied water coverage is limited to downloaded counties.",
                    "When supplied, land-chord checks are conservative: they may reject genuine bridge or indirect land paths.",
                    "Pickup/drop-off types 1, 2 and 3 are unavailable to unrestricted riders in this model.",
                    "Interlining, timed transfers, schedule correlations, crowding, service reliability, and accessibility are not modeled.",
                    "GTFS minimum-transfer rules conservatively delay all onward transfers from affected arrival states; prohibitions disable all onward transfers there. This prevents walking via a third stop from bypassing a rule, but can overestimate journeys unaffected by the original rule.",
                    "Transit requires at least one actual GTFS ride edge; absent service stays infinite; no invented transit fallback."]}
    manifest_path.write_text(json.dumps(manifest, indent=2)+"\n")
    print(f"  Saved travel matrices for {n:,} tracts", flush=True)
    return manifest
