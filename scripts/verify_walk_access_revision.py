"""Independently check saved walking-access results; no routing or network calls.

Run after all cities and aggregate tables exist. Add --require-maps after the map
gallery is finished. Only verification.json is written; source results are read.
The defining equations below intentionally do not call the production scorers.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from zipfile import ZipFile

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/multicity"
OUTPUT = BASE / "walk_access_revision"
CITIES = ("philadelphia", "miami", "los_angeles", "salt_lake_city",
          "san_francisco", "washington_dc", "houston", "detroit", "chicago",
          "phoenix", "atlanta", "new_york_city")
GROUPS = ("white_alone", "black_alone", "hispanic_latino_any_race", "asian_alone")
SCENARIOS = ("with_transit", "walking_only")
METRICS = ("segregation_index", "segregation_index_minus_K", "segregation_index_normalized")
ECONOMICS = {"beta_per_usd": .1, "omega_usd_per_hour": 15., "c0_usd": 1.25,
             "standardized_transit_fare_usd": 3.}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, message, atol=1e-11):
    try:
        np.testing.assert_allclose(actual, expected, atol=atol, rtol=0, equal_nan=True)
    except AssertionError as error:
        raise ValueError(message + ": " + str(error)) from error


def equal(actual, expected, message):
    try:
        np.testing.assert_array_equal(actual, expected)
    except AssertionError as error:
        raise ValueError(message + ": " + str(error)) from error


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def read_csv(path):
    return pd.read_csv(path, dtype={"tract_geoid": str, "node_id": str,
                                   "section_id": str, "zone_id": str})


def read_npz(path):
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


def hashes(entries, verified):
    records = entries.items() if isinstance(entries, dict) else ((r["path"], r["sha256"]) for r in entries)
    for name, expected in records:
        path = Path(name)
        path = path if path.is_absolute() else ROOT / path
        if path not in verified:
            verified[path] = sha256(path)
        require(verified[path] == expected, f"Source checksum mismatch: {name}")


def original_nodes(city):
    if city != "new_york_city":
        return read_csv(BASE / city / "demographics/nodes.csv")
    z = read_csv(ROOT / "outputs/nyc_routes/zones.csv")
    return pd.DataFrame({"matrix_index": z.matrix_index, "tract_geoid": z.zone_id,
                         "latitude": z.latitude, "longitude": z.longitude,
                         "population": z.TOTPOP, "white_alone": z.NH_WHITE + z.H_WHITE,
                         "black_alone": z.NH_BLACK + z.H_BLACK,
                         "hispanic_latino_any_race": z.HISP,
                         "asian_alone": z.NH_ASIAN + z.H_ASIAN})


def selected_stop_coordinates(travel, identifiers, verified):
    """Read saved GTFS stop coordinates, including its parent-coordinate fallback."""
    lookup = {}
    for feed_number, feed in enumerate(travel["feeds"]):
        path = Path(feed["path"])
        path = path if path.is_absolute() else ROOT / path
        hashes({str(path): feed["sha256"]}, verified)
        with ZipFile(path) as archive:
            matches = [name for name in archive.namelist() if Path(name).name == "stops.txt"]
            require(len(matches) == 1, f"Ambiguous/missing stops table: {path}")
            stops = pd.read_csv(archive.open(matches[0]), dtype=str, keep_default_na=False)
        stops.columns = stops.columns.str.strip()
        for column in stops:
            stops[column] = stops[column].str.strip()
        require(stops.stop_id.is_unique, f"Duplicate GTFS stop IDs: {path}")
        for field in ("stop_lat", "stop_lon"):
            stops[field] = pd.to_numeric(stops[field], errors="coerce")
            if "parent_station" in stops:
                parents = stops.set_index("stop_id")[field]
                stops[field] = stops[field].fillna(stops.parent_station.map(parents))
        for row in stops.itertuples():
            lookup[f"{feed_number}:{row.stop_id}"] = (row.stop_lat, row.stop_lon)
    require(len(set(identifiers)) == len(identifiers), "Duplicate saved transit stop IDs")
    require(all(str(s) in lookup for s in identifiers), "Saved transit stop missing from cached GTFS")
    coords = np.array([lookup[str(s)] for s in identifiers], dtype=float).reshape(-1, 2)
    require(np.isfinite(coords).all(), "Missing coordinates for a used transit stop")
    return coords


def selected_walk_seconds(left, right, parameters):
    """Independent vectorized haversine for corresponding endpoint pairs."""
    a, b = np.radians(left), np.radians(right)
    h = np.sin((b[:, 0] - a[:, 0]) / 2) ** 2
    h += np.cos(a[:, 0]) * np.cos(b[:, 0]) * np.sin((b[:, 1] - a[:, 1]) / 2) ** 2
    km = 2 * 6371.0088 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))
    return km * parameters["walking_detour"] * 3600 / parameters["walking_speed_kmh"] + parameters["entrance_seconds"]


def verify_travel(city, nodes, modes, travel, verified):
    n = len(nodes)
    p = travel["parameters"]
    require(p["access_policy"] == "uncapped" and p["pedestrian_barrier_policy"] == "reject", f"{city}: wrong access policy")
    close([p["walking_speed_kmh"], p["walking_detour"], p["transfer_walk_km"],
           p["entrance_seconds"], p["transfer_buffer_seconds"]], [4.8, 1.3, .4, 60, 60], f"{city}: changed walking parameters", 0)
    equal(modes["tract_geoid"], nodes.tract_geoid, f"{city}: travel/node alignment")
    equal(modes["matrix_index"], np.arange(n), f"{city}: travel matrix index")
    walk, transit = modes["walk_seconds"], modes["transit_seconds"]
    require(walk.shape == transit.shape == (n, n), f"{city}: travel shape")
    require(np.isfinite(walk).all() and (walk >= 0).all(), f"{city}: incomplete walking matrix")
    require(not np.isnan(transit).any() and (transit >= 0).all(), f"{city}: invalid transit times")
    require(np.isinf(np.diag(transit)).all(), f"{city}: transit diagonal must be unavailable")
    close(np.diag(walk), 0, f"{city}: walking diagonal", 0)
    close(walk, walk.T, f"{city}: shared walking symmetry", 1e-8)
    finite = np.isfinite(transit)
    # These fixed twelve cached networks all serve populated city locations.
    # A wholly empty candidate matrix is a pipeline failure to investigate,
    # not evidence that removing transit has no effect (e.g. an empty land mask).
    require(finite.any(), f"{city}: no transit candidates; inspect loaded geometry and boarding access")
    require((modes["access_stop_count"] > 0).any(), f"{city}: no origin has boarding access; inspect geometry coverage")
    stops = selected_stop_coordinates(travel, modes["transit_stop_ids"], verified)
    coords = nodes[["latitude", "longitude"]].to_numpy()
    ns = len(stops)
    for i in range(n):
        expected = selected_walk_seconds(np.repeat(coords[[i]], n, axis=0), coords, p) - p["entrance_seconds"]
        close(walk[i], expected, f"{city}: direct walking distance formula", 1e-6)
    maximum_walk_error = 0.
    for role, row_end in (("access", True), ("egress", False)):
        index_key = "transit_first_boarding_stop_index" if row_end else "transit_last_alighting_stop_index"
        idx = modes[index_key]
        times = modes[f"transit_{role}_seconds"]
        require(idx.shape == times.shape == (n, n), f"{city}: {role} diagnostics shape")
        require(np.issubdtype(idx.dtype, np.integer), f"{city}: {role} stop indexes are not integers")
        require(((idx[finite] >= 0) & (idx[finite] < ns)).all(), f"{city}: invalid selected {role} stop")
        require((idx[~finite] == -1).all(), f"{city}: unavailable journey has a selected {role} stop")
        require(np.isfinite(times[finite]).all() and np.isinf(times[~finite]).all(), f"{city}: {role} availability differs")
        require(not modes[f"transit_{role}_barrier_flag"].any(), f"{city}: rejected barrier leg was retained")
        for i in range(n):
            js = np.flatnonzero(finite[i])
            if not len(js):
                continue
            endpoints = np.repeat(coords[[i]], len(js), axis=0) if row_end else coords[js]
            expected = selected_walk_seconds(endpoints, stops[idx[i, js]], p)
            close(times[i, js], expected, f"{city}: selected {role} time excludes/changes walking", 1e-6)
            maximum_walk_error = max(maximum_walk_error, float(np.max(np.abs(times[i, js] - expected))))
    charged = modes["transit_access_seconds"][finite] + modes["transit_egress_seconds"][finite]
    require((transit[finite] + 1e-7 >= charged).all(), f"{city}: access/egress not fully charged")
    require((modes["access_stop_count"] >= 0).all() and (modes["access_stop_count"] <= ns).all(), f"{city}: invalid access counts")
    require(not finite[modes["access_stop_count"] == 0].any(), f"{city}: journey from origin with no boarding access")
    old_feasible = 0
    if city != "new_york_city":
        old = read_npz(BASE / city / "travel/modes.npz")
        old_manifest = read_json(BASE / city / "travel/manifest.json")
        equal(old["tract_geoid"], nodes.tract_geoid, f"{city}: old/new node order")
        close(walk, old["walk_seconds"], f"{city}: walking baseline changed", 1e-8)
        equal(modes["walk_land_supported"], old["walk_land_supported"], f"{city}: walking barrier mask changed")
        require([(f["sha256"], f["service_date"]) for f in travel["feeds"]] ==
                [(f["sha256"], f["service_date"]) for f in old_manifest["feeds"]], f"{city}: schedules changed")
        for field, value in old_manifest["parameters"].items():
            require(p[field] == value, f"{city}: original parameter changed: {field}")
        for field in ("land_sha256", "water_sha256"):
            require(travel["input_fingerprint"].get(field) == old_manifest["input_fingerprint"].get(field), f"{city}: barrier geometry changed")
        before = np.isfinite(old["transit_seconds"])
        require((transit[before] <= old["transit_seconds"][before] + 1e-6).all(), f"{city}: uncapped transit worsened a feasible old candidate")
        old_feasible = int(before.sum())
    else:
        require(sum(f["used_routes_by_mode"]["bus"] for f in travel["feeds"]) == 0, "NYC unexpectedly includes bus routes")
        require(sum(f["used_routes_by_mode"]["rail"] for f in travel["feeds"]) == 29, "NYC subway/SIR coverage differs")
    return {"transit_finite_pairs": int(finite.sum()), "old_finite_candidates_preserved": old_feasible,
            "maximum_access_egress_reconstruction_error_seconds": maximum_walk_error}


def verify_scenario(city, scenario, nodes, modes, scores, local, verified):
    folder = OUTPUT / city / scenario
    costs, arrays = read_npz(folder / "cost_matrices.npz"), read_npz(folder / "analysis_matrices.npz")
    report = read_json(folder / "manifest.json")
    require(report["parameters"] == ECONOMICS, f"{city}: economics changed")
    require(report["routes_api_calls"] == 0 and report["no_transit_proxy"], f"{city}: unsupported routing provenance")
    hashes(report["source_sha256"], verified)
    n = len(nodes)
    D = nodes.population.to_numpy(dtype=float)
    counts = nodes[list(GROUPS)].to_numpy(dtype=np.int64)
    proportions = counts / D[:, None]
    for saved in (costs, arrays):
        equal(saved["tract_geoid"], nodes.tract_geoid, f"{city}/{scenario}: matrix/node order")
    equal(arrays["group_names"], GROUPS, f"{city}/{scenario}: group order")
    equal(arrays["population"], D, f"{city}/{scenario}: population")
    equal(arrays["counts"], counts, f"{city}/{scenario}: category counts")
    close(arrays["proportions"], proportions, f"{city}/{scenario}: proportions", 1e-14)
    expected_walk = modes["walk_seconds"] * 15 / 3600
    expected_transit = modes["transit_seconds"] * 15 / 3600 + 3
    if scenario == "walking_only":
        expected_transit = np.full((n, n), np.inf)
    C = np.maximum(1.25, np.minimum(expected_walk, expected_transit))
    np.fill_diagonal(C, 1.25)
    F = np.exp(-.1 * C)
    selected = np.where(expected_walk <= expected_transit, "WALK", "TRANSIT")
    np.fill_diagonal(selected, "LOCAL")
    for key, value in (("walk_cost_usd", expected_walk), ("transit_cost_usd", expected_transit),
                       ("C_usd", C), ("log_F", -.1 * C), ("F", F)):
        close(costs[key], value, f"{city}/{scenario}: {key} defining formula", 1e-11)
    equal(costs["chosen_mode"], selected, f"{city}/{scenario}: mode minimum/tie choice")
    flags = ~modes["walk_land_supported"].copy() if bool(modes["walk_barrier_check_available"]) else np.zeros((n, n), bool)
    np.fill_diagonal(flags, False)
    equal(costs["walk_barrier_flag"], flags, f"{city}/{scenario}: water/land flags")
    equal(costs["chosen_walk_barrier_flag"], flags & (selected == "WALK"), f"{city}/{scenario}: selected walking flags")
    P = F * D[None, :]
    P /= P.sum(axis=1)[:, None]
    close(arrays["P"], P, f"{city}/{scenario}: P defining formula", 1e-12)
    close(arrays["P"].sum(axis=1), 1, f"{city}/{scenario}: transition row sums", 1e-12)
    require(arrays["A"].shape == (4, n, n), f"{city}/{scenario}: A dimensions")
    require(arrays["right_eigenvectors"].shape == (n, 4), f"{city}/{scenario}: eigenvector dimensions")
    local_disk = read_csv(folder / "tract_scores.csv")
    summary_disk = read_csv(folder / "segregation_comparison.csv")
    # Concatenating tract and NYC-section exports adds the tract-only fields to
    # NYC rows as empty columns. Validate those extras, then compare the actual
    # per-city schema in its saved order instead of requiring identical widths.
    require(set(local_disk.columns).issubset(local.columns), f"{city}/{scenario}: aggregate local table omitted columns")
    extra_columns = [column for column in local if column not in local_disk]
    require(local[extra_columns].isna().all().all(), f"{city}/{scenario}: aggregate-only local columns contain data")
    pd.testing.assert_frame_equal(local[local_disk.columns].reset_index(drop=True), local_disk,
                                  check_dtype=False, atol=1e-12, rtol=0)
    pd.testing.assert_frame_equal(scores.reset_index(drop=True), summary_disk, check_dtype=False, atol=1e-12, rtol=0)
    maximum_residual = 0.
    maximum_collatz_width = 0.
    for r, group in enumerate(GROUPS):
        row = scores.loc[scores.group.eq(group)].iloc[0]
        loc = local.loc[local.group.eq(group)].sort_values("matrix_index")
        equal(loc.tract_geoid, nodes.tract_geoid, f"{city}/{scenario}/{group}: local ordering")
        equal(loc.matrix_index, np.arange(n), f"{city}/{scenario}/{group}: local matrix index")
        p = proportions[:, r]
        K = counts[:, r].sum() / D.sum()
        A = P * p[None, :]
        close(arrays["A"][r], A, f"{city}/{scenario}/{group}: A defining formula", 1e-12)
        v, value = arrays["right_eigenvectors"][:, r], float(arrays["eigenvalues"][r])
        require(np.isfinite(v).all() and (v > 0).all(), f"{city}/{scenario}/{group}: nonpositive Perron vector")
        require(0 <= value <= 1 + 1e-12, f"{city}/{scenario}/{group}: invalid city eigenvalue")
        close(v.sum(), 1, f"{city}/{scenario}/{group}: eigenvector normalization", 1e-12)
        product = A @ v
        residual = float(np.max(np.abs(product - value * v)))
        require(residual < 1e-10, f"{city}/{scenario}/{group}: large eigenpair residual {residual}")
        ratios = product / v
        require(ratios.min() - 1e-10 <= value <= ratios.max() + 1e-10, f"{city}/{scenario}/{group}: eigenvalue outside Collatz bounds")
        maximum_residual = max(maximum_residual, residual)
        maximum_collatz_width = max(maximum_collatz_width, float(np.ptp(ratios)))
        expected = {"population": counts[:, r].sum(), "population_share_K": K, "population_percent": 100*K,
                    "segregation_index": value, "segregation_index_minus_K": value-K,
                    "segregation_index_normalized": (value-K)/(1-K), "city_population": D.sum(), "node_count": n}
        weights = counts[:, r] / counts[:, r].sum()
        residential, network = float(weights @ p), float(weights @ (P @ p))
        expected.update(residential_isolation=residential, residential_isolation_normalized=(residential-K)/(1-K),
                        network_isolation=network, network_isolation_normalized=(network-K)/(1-K),
                        dissimilarity_vs_rest=.5*np.abs(weights-(D-counts[:, r])/(D.sum()-counts[:, r].sum())).sum())
        for name, expected_value in expected.items():
            close(row[name], expected_value, f"{city}/{scenario}/{group}: {name}", 1e-11)
        local_expected = {"population": D, "group_population": counts[:, r], "group_proportion": p,
                          "population_share_K": K, "pssi_city_eigenvalue": value,
                          "pssi_right_eigenvector": v, "pssi_eigenvector_times_n": n*v,
                          "network_same_group_exposure": P @ p}
        for name, expected_value in local_expected.items():
            close(loc[name], expected_value, f"{city}/{scenario}/{group}: local {name}", 1e-11)
        close(row.eigenpair_max_abs_residual, residual, f"{city}/{scenario}/{group}: reported residual", 1e-11)
    return {"max_pssi_eigenpair_residual": maximum_residual,
            "max_collatz_perron_interval_width": maximum_collatz_width,
            "transit_selected_pairs": int((selected == "TRANSIT").sum())}, C


def verify_fryer(paired, verified):
    fresh = read_csv(OUTPUT / "distance_fryer_recomputed.csv")
    old = read_csv(ROOT / "outputs/distance_fryer/scores_all_radii.csv")
    old = old.loc[old.radius_km.eq(1)].set_index(["city_id", "node_scale", "group"])
    require(len(fresh) == 96 and not fresh.duplicated(["city_id", "fryer_node_scale", "group"]).any(), "Fryer result completeness")
    hashes(read_json(OUTPUT / "distance_fryer_sources.json"), verified)
    max_change = 0.
    undefined = 0
    for city in CITIES:
        for scale in ("block", "tract"):
            nodes = read_csv(ROOT / "outputs/distance_fryer" / city / scale / "nodes.csv")
            D = nodes.population.sum()
            for group in GROUPS:
                rows = fresh.loc[fresh.city_id.eq(city) & fresh.fryer_node_scale.eq(scale) & fresh.group.eq(group)]
                require(len(rows) == 1, f"Missing Fryer row {city}/{scale}/{group}")
                row, previous = rows.iloc[0], old.loc[(city, scale, group)]
                K = nodes[group].sum() / D
                close(row.population_share_K, K, f"{city}/{scale}/{group}: Fryer K", 1e-14)
                close(row.population_share_K, previous.population_share_K, f"{city}/{scale}/{group}: Fryer K changed", 1e-14)
                close(row.fryer_ssi, previous.fryer_ssi, f"{city}/{scale}/{group}: Fryer changed", 1e-10)
                close(row.fryer_ssi_minus_K, row.fryer_ssi-K, f"{city}/{scale}/{group}: SSI-K", 1e-12)
                close(row.fryer_ssi_normalized, (row.fryer_ssi-K)/(1-K), f"{city}/{scale}/{group}: rescaled SSI", 1e-12)
                equal(row.eligible_assigned_node_count, previous.eligible_assigned_node_count, "Fryer denominator changed")
                require(row.status == previous.status and row.fryer_node_count == len(nodes), "Fryer status/node count changed")
                require(row.eigenpair_max_abs_residual <= 1e-10, "Large recomputed Fryer eigenpair residual")
                close(row.change_from_previous, row.fryer_ssi-previous.fryer_ssi, "Fryer reported change", 1e-12)
                if np.isfinite(row.fryer_ssi):
                    max_change = max(max_change, abs(row.fryer_ssi-previous.fryer_ssi))
                else:
                    undefined += 1
                    require(row.eligible_assigned_node_count == 0, "Undefined Fryer has eligible nodes")
                wide = paired.loc[paired.city_id.eq(city) & paired.group.eq(group)].iloc[0]
                for field in ("fryer_ssi", "fryer_ssi_minus_K", "fryer_ssi_normalized", "fryer_node_count"):
                    close(wide[f"distance_{scale}_{field}"], row[field], "Wide Fryer table differs", 1e-12)
                require(wide[f"distance_{scale}_status"] == row.status, "Wide Fryer status differs")
    return {"rows": len(fresh), "undefined_rows": undefined, "maximum_change_from_previous": max_change}


def verify_maps(verified, required=False, skip=False):
    folder = OUTPUT / "maps"
    if skip or not folder.exists():
        require(not required, "Map gallery is required but absent/skipped")
        return {"checked": False, "reason": "explicitly skipped" if skip else "not yet generated"}
    from PIL import Image
    images = []
    for city in CITIES:
        target = folder / city
        metadata = read_json(target / "metadata.json")
        sources = metadata["sources"]
        hashes(sources, verified)
        expected_source = str((OUTPUT / city / "with_transit/analysis_matrices.npz").relative_to(ROOT))
        require(expected_source in {r["path"] for r in sources}, f"{city}: map lacks revised analysis hash")
        nodes = read_csv(OUTPUT / city / "nodes.csv")
        arrays = read_npz(OUTPUT / city / "with_transit/analysis_matrices.npz")
        data = read_csv(target / "section_eigenvectors.csv")
        require(metadata["city_id"] == city and metadata["section_count"] == len(nodes), f"{city}: map metadata alignment")
        require(metadata["population"] == int(nodes.population.sum()), f"{city}: map population")
        require(len(data) == len(nodes)*4, f"{city}: map CSV completeness")
        for r, group in enumerate(GROUPS):
            values = data.loc[data.category.eq(group)]
            equal(values.section_id, nodes.tract_geoid, f"{city}/{group}: map node alignment")
            equal(values.matrix_index, nodes.matrix_index, f"{city}/{group}: map matrix order")
            for field, expected in (("raw_eigenvector_v", arrays["right_eigenvectors"][:, r]),
                                    ("relative_eigenvector_n_v", len(nodes)*arrays["right_eigenvectors"][:, r]),
                                    ("citywide_eigenvalue_lambda", arrays["eigenvalues"][r]),
                                    ("category_population", arrays["counts"][:, r]),
                                    ("category_proportion", arrays["proportions"][:, r]),
                                    ("population", nodes.population)):
                close(values[field], expected, f"{city}/{group}: map {field}", 1e-11)
            relative = len(nodes)*arrays["right_eigenvectors"][:, r]
            low, high = metadata["color_scale"]["minimum"], metadata["color_scale"]["maximum"]
            equal(values.below_color_scale, relative < low, f"{city}/{group}: map low-color clipping flags")
            equal(values.above_color_scale, relative > high, f"{city}/{group}: map high-color clipping flags")
            diagnostic = metadata["category_diagnostics"][group]
            close(diagnostic["vector_sum"], 1., "Map eigenvector sum", 1e-12)
            close(diagnostic["relative_score_mean"], 1., "Map relative-score mean", 1e-12)
            close(diagnostic["minimum"], relative.min(), "Map minimum", 1e-12)
            close(diagnostic["maximum"], relative.max(), "Map maximum", 1e-11)
            require(diagnostic["below_color_scale"] == int((relative < low).sum()) and
                    diagnostic["above_color_scale"] == int((relative > high).sum()), "Map saturation counts")
        names = [f"{group}_pssi_eigenvector.png" for group in GROUPS] + ["four_category_pssi_eigenvectors.png"]
        require({p.name for p in target.glob("*.png")} == set(names), f"{city}: expected five map PNGs")
        for name in names:
            path = target / name
            with Image.open(path) as picture:
                require(picture.format == "PNG" and min(picture.size) > 100, f"Invalid map PNG: {path}")
                picture.verify()
            images.append(path)
    gallery = read_json(folder / "manifest.json")
    require(gallery["city_count"] == 12 and gallery["png_count"] == 60 and len(images) == 60, "Map gallery count mismatch")
    return {"checked": True, "cities": 12, "png_images": 60,
            "checks": "PNG integrity; source SHA256 correspondence; exact node ordering, eigenvectors and n*v exports"}


def verify_previous_reference(paired):
    """The eleven-city old/new comparison is distinct from the paired toggle."""
    reference = read_csv(OUTPUT / "previous_results_comparison.csv").set_index(["city_id", "group"])
    previous = read_csv(BASE / "transit_comparison/scenario_scores.csv")
    previous = previous.loc[previous.scenario.eq("with_transit")].set_index(["city_id", "group"])
    require(len(reference) == 48 and reference.index.is_unique, "Previous-results reference completeness")
    for row in paired.itertuples():
        key = (row.city_id, row.group)
        ref = reference.loc[key]
        for metric in METRICS:
            current = getattr(row, metric+"_with_transit")
            close(ref[metric+"_with_transit"], current, "Previous-results reference uses wrong new score", 1e-12)
            if row.city_id == "new_york_city":
                require(pd.isna(ref[metric+"_previous_capped"]) and pd.isna(ref[metric+"_change_new_minus_old"]),
                        "NYC must not claim a comparable capped multicity baseline")
            else:
                old = previous.loc[key, metric]
                close(ref[metric+"_previous_capped"], old, "Previous-results reference uses wrong old score", 1e-12)
                close(ref[metric+"_change_new_minus_old"], current-old, "Previous-results change sign/value", 1e-12)


def verify(*, require_maps=False, skip_maps=False):
    scores, paired = read_csv(OUTPUT / "scenario_scores.csv"), read_csv(OUTPUT / "city_scores.csv")
    local = read_csv(OUTPUT / "all_node_scores.csv")
    require(len(scores) == 96 and len(paired) == 48 and len(local) == 38464, "Expected 96 scenario, 48 paired and 38464 local rows")
    require(set(scores.city_id) == set(paired.city_id) == set(CITIES), "Expected all 12 cities")
    require(set(scores.scenario) == set(SCENARIOS) and set(scores.group) == set(GROUPS), "Unexpected scenarios/categories")
    require(not scores.duplicated(["city_id", "scenario", "group"]).any(), "Duplicate scenario scores")
    require(not paired.duplicated(["city_id", "group"]).any(), "Duplicate paired scores")
    require(not local.duplicated(["city_id", "scenario", "group", "tract_geoid"]).any(), "Duplicate local rows")
    verified, city_reports = {}, []
    max_residual, total_nodes = 0., 0
    for city in CITIES:
        print(f"Verifying {city}", flush=True)
        folder = OUTPUT / city
        nodes = read_csv(folder / "nodes.csv")
        original = original_nodes(city)
        n = len(nodes)
        require(nodes.tract_geoid.is_unique, f"{city}: duplicate nodes")
        equal(nodes.matrix_index, np.arange(n), f"{city}: node ordering")
        for field in ("tract_geoid", "matrix_index", "population", *GROUPS):
            equal(nodes[field], original[field], f"{city}: original {field} changed")
        close(nodes[["latitude", "longitude"]], original[["latitude", "longitude"]], f"{city}: original coordinates changed", 1e-13)
        require((nodes.population > 0).all(), f"{city}: nonpopulated node")
        total_nodes += n
        modes = read_npz(folder / "travel/modes.npz")
        travel = read_json(folder / "travel/manifest.json")
        report = {"city_id": city, "node_count": n, **verify_travel(city, nodes, modes, travel, verified)}
        prior_C = None
        report["scenarios"] = {}
        for scenario in SCENARIOS:
            selected_scores = scores.loc[scores.city_id.eq(city) & scores.scenario.eq(scenario)]
            selected_local = local.loc[local.city_id.eq(city) & local.scenario.eq(scenario)]
            require(len(selected_scores) == 4 and len(selected_local) == n*4, f"{city}/{scenario}: incomplete rows")
            info, C = verify_scenario(city, scenario, nodes, modes, selected_scores, selected_local, verified)
            report["scenarios"][scenario] = info
            max_residual = max(max_residual, info["max_pssi_eigenpair_residual"])
            if prior_C is not None:
                require((prior_C <= C+1e-11).all(), f"{city}: removing transit reduced generalized cost")
            prior_C = C
        city_reports.append(report)
    require(total_nodes == 4808 and total_nodes*8 == len(local), "Unexpected city node totals")
    lookup = scores.set_index(["city_id", "scenario", "group"])
    for row in paired.itertuples():
        for scenario in SCENARIOS:
            source = lookup.loc[(row.city_id, scenario, row.group)]
            for field in ("node_count", "city_population", "population", "population_share_K"):
                close(getattr(row, field), source[field], "Paired demographic column differs", 1e-12)
            for metric in METRICS:
                close(getattr(row, metric+"_"+scenario), source[metric], "Paired score differs", 1e-12)
        for metric in METRICS:
            close(getattr(row, metric+"_walking_minus_transit"),
                  getattr(row, metric+"_walking_only")-getattr(row, metric+"_with_transit"), "Paired difference sign/value", 1e-12)
    verify_previous_reference(paired)
    fryer = verify_fryer(paired, verified)
    maps = verify_maps(verified, required=require_maps, skip=skip_maps)
    for name, frame in (("score_column_dictionary.csv", scores), ("node_column_dictionary.csv", local), ("city_column_dictionary.csv", paired)):
        dictionary = read_csv(OUTPUT / name)
        require(dictionary.column.is_unique and set(dictionary.column) == set(frame.columns), f"Incomplete dictionary: {name}")
        require(dictionary.definition.notna().all(), f"Missing column definitions: {name}")
    manifest = read_json(OUTPUT / "manifest.json")
    require(manifest["routes_api_calls"] == 0 and manifest["network_requests_during_calculation"] == 0, "Reported network calls")
    require(manifest["parameters"] == ECONOMICS, "Top-level parameters differ")
    report = {"passed": True, "status": "passed", "verified_at_utc": datetime.now(timezone.utc).isoformat(),
              "cities": 12, "scenario_category_rows": len(scores), "paired_category_rows": len(paired),
              "nodes_per_scenario": total_nodes, "local_rows_both_scenarios": len(local),
              "maximum_pssi_eigenpair_residual": max_residual, "source_files_checksum_verified": len(verified),
              "city_checks": city_reports, "distance_fryer": fryer, "maps": maps,
              "checks": ["Unchanged original Census counts, coordinates and node ordering",
                         "Cached feed checksums, service dates and fixed controlled-comparison parameters",
                         "Selected boarding/alighting indexes and full walking-leg times reconstructed from GTFS stop coordinates",
                         "Uncapped transit preserves every formerly feasible candidate; identical direct walking baseline",
                         "Generalized costs, mode choices, F, P and all four A matrices rebuilt from definitions",
                         "Positive sum-one eigenvectors, eigenpair residuals, local exports and exposure statistics",
                         "Population-share adjustments and exact signed paired differences",
                         "Geographic Fryer scores unchanged, with undefined groups retained as NaN",
                         "Complete column dictionaries and optional map provenance/integrity"],
              "limits": ["This verifier does not claim walking chords are real street/bridge itineraries.",
                         "It does not independently rerun all GTFS shortest paths; ride-required topology and optimality were separately checked with independent synthetic shortest-path oracles.",
                         "Source hashes verify provenance, not schedule accuracy or real-time service."],
              "routes_api_calls_during_verification": 0, "network_calls_during_verification": 0}
    (OUTPUT / "verification.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("city_checks", "checks", "limits")}, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    options = parser.add_mutually_exclusive_group()
    options.add_argument("--require-maps", action="store_true", help="Require the finished 12-city, 60-image map gallery")
    options.add_argument("--skip-maps", action="store_true", help="Verify numerical results before maps are generated")
    args = parser.parse_args()
    try:
        verify(require_maps=args.require_maps, skip_maps=args.skip_maps)
    except Exception as error:
        failure = {"passed": False, "status": "incomplete" if isinstance(error, FileNotFoundError) else "failed",
                   "verified_at_utc": datetime.now(timezone.utc).isoformat(), "error": str(error),
                   "error_type": type(error).__name__, "network_calls_during_verification": 0}
        OUTPUT.mkdir(parents=True, exist_ok=True)
        (OUTPUT / "verification.json").write_text(json.dumps(failure, indent=2)+"\n")
        print(json.dumps(failure, indent=2), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
