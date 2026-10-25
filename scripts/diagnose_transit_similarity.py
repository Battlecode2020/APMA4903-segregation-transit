"""Diagnose saved transit-on/off spectral similarities without routing or downloads.

Reads the controlled comparison's identical-walking scenarios. Static GTFS is
read only to locate active boardable bus/rail stops; no transit paths are
recomputed. Writes only outputs/multicity/transit_diagnostics by default.
"""
from __future__ import annotations

import argparse
import hashlib
from html import escape
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from scipy.spatial import cKDTree

from scripts.multicity_analysis import perron_pair
from scripts.multicity_transit import TransitParameters, _point_distances, _walk_validity, _xyz, load_patterns


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/multicity"
COMPARISON = BASE / "transit_comparison"
OUTPUT = BASE / "transit_diagnostics"


def read_npz(path):
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


def access_diagnostics(city, nodes, modes, travel):
    """Reproduce stop-access screening, never compute any journey or route."""
    from datetime import date

    parameters = TransitParameters(**travel["parameters"])
    stop_frames = []
    for number, feed in enumerate(travel["feeds"]):
        path = Path(feed["path"])
        if not path.is_absolute():
            path = ROOT / path
        if hashlib.sha256(path.read_bytes()).hexdigest() != feed["sha256"]:
            raise ValueError(f"Feed changed since saved routing: {path}")
        stops, patterns, _, _ = load_patterns(path, date.fromisoformat(feed["service_date"]), parameters)
        boardable = {mode: set() for mode in ("bus", "rail")}
        all_modes = {mode: set() for mode in ("bus", "rail")}
        for pattern in patterns:
            all_modes[pattern["mode"]].update(pattern["stop_ids"])
            for k, stop_id in enumerate(pattern["stop_ids"][:-1]):
                if pattern["pickup"][k] and np.isfinite(pattern["boarding_wait_seconds"][k]):
                    boardable[pattern["mode"]].add(stop_id)
        # Match the saved router's stop-level radius: a physical stop that has
        # any active rail pattern gets the rail radius, even if it also has bus.
        selected = stops.loc[stops.stop_id.isin(boardable["bus"] | boardable["rail"])].copy()
        selected["boardable_bus"] = selected.stop_id.isin(boardable["bus"])
        selected["boardable_rail"] = selected.stop_id.isin(boardable["rail"])
        selected["radius_mode"] = np.where(selected.stop_id.isin(all_modes["rail"]), "rail", "bus")
        selected["feed_id"] = path.stem
        selected["stop_key"] = str(number) + ":" + selected.stop_id
        stop_frames.append(selected)
    stops = pd.concat(stop_frames, ignore_index=True)
    coordinates = nodes[["latitude", "longitude"]].to_numpy(dtype=float)
    stopcoords = stops[["stop_lat", "stop_lon"]].to_numpy(dtype=float)
    stop_tree = cKDTree(_xyz(stopcoords))
    nearest = {}
    for mode in ("bus", "rail"):
        eligible = np.flatnonzero(stops["boardable_" + mode].to_numpy())
        if len(eligible):
            _, positions = cKDTree(_xyz(stopcoords[eligible])).query(_xyz(coordinates), k=1)
            closest = eligible[positions]
            nearest[mode] = {
                "distance": _point_distances(coordinates, stopcoords[closest]),
                "stop_id": stops.stop_key.iloc[closest].to_numpy(),
                "stop_name": stops.stop_name.iloc[closest].to_numpy() if "stop_name" in stops else np.full(len(nodes), ""),
            }
        else:
            nearest[mode] = {"distance": np.full(len(nodes), np.inf), "stop_id": np.full(len(nodes), ""), "stop_name": np.full(len(nodes), "")}
    water = gpd.read_file(BASE/city/"demographics/water.geojson").geometry.union_all()
    shapely.prepare(water)
    radius_by_stop = np.where(stops.radius_mode.eq("rail"), parameters.rail_access_km, parameters.bus_access_km)
    rows = []
    for i, point in enumerate(coordinates):
        candidate = np.asarray(stop_tree.query_ball_point(_xyz(coordinates[[i]])[0], max(parameters.bus_access_km, parameters.rail_access_km)/parameters.walking_detour), dtype=int)
        if len(candidate):
            distances = _point_distances(np.repeat(point[None, :], len(candidate), axis=0), stopcoords[candidate])
            candidate = candidate[distances*parameters.walking_detour <= radius_by_stop[candidate]]
        prewater_count = len(candidate)
        postwater_count = int(_walk_validity(np.repeat(point[None, :], prewater_count, axis=0), stopcoords[candidate], water_geometry=water).sum()) if prewater_count else 0
        saved_count = int(modes["access_stop_count"][i])
        if postwater_count != saved_count:
            raise ValueError(f"Access screening does not reproduce saved count: {city} {nodes.tract_geoid[i]} {postwater_count}!={saved_count}")
        if saved_count:
            reason = "has_modeled_boarding_access"
        elif not prewater_count:
            reason = "no_boardable_stop_within_modeled_radius"
        else:
            reason = "all_nearby_boardable_stops_rejected_by_water_chord_check"
        row = {"city_id": city, **nodes.iloc[i].to_dict(), "saved_access_stop_count": saved_count,
               "boardable_stop_candidates_before_water_check": prewater_count,
               "boardable_stop_candidates_after_water_check": postwater_count,
               "modeled_access_status": reason,
               "transit_reachable_destinations": int(np.isfinite(modes["transit_seconds"][i]).sum()),
               "transit_reachable_origins": int(np.isfinite(modes["transit_seconds"][:, i]).sum()),
               "nearest_other_tract_geodesic_km": float(np.min(np.delete(modes["great_circle_distance_km"][i], i)))}
        for mode in ("bus", "rail"):
            row[f"nearest_active_{mode}_stop_geodesic_km"] = float(nearest[mode]["distance"][i])
            row[f"nearest_active_{mode}_stop_walk_proxy_km"] = parameters.walking_detour*float(nearest[mode]["distance"][i])
            row[f"nearest_active_{mode}_stop_key"] = str(nearest[mode]["stop_id"][i])
            row[f"nearest_active_{mode}_stop_name"] = str(nearest[mode]["stop_name"][i])
        rows.append(row)
    return pd.DataFrame(rows)


def spectral_details(A, value, vector):
    left_value, left, residual, _ = perron_pair(A.T)
    if abs(left_value-value) > 1e-9:
        raise ValueError("Left and right eigenvalues disagree")
    product = float(left @ vector)
    if product <= 0:
        raise ValueError("Invalid Perron vector overlap")
    elasticity = left*vector/product
    if not np.isclose(elasticity.sum(), 1, atol=1e-12):
        raise ValueError("Invalid row-elasticity normalization")
    return left, elasticity, residual


def diagnose_city(city, top_count=10):
    folder = BASE/city
    nodes = pd.read_csv(folder/"demographics/nodes.csv", dtype={"tract_geoid": str})
    modes = read_npz(folder/"travel/modes.npz")
    travel = json.loads((folder/"travel/manifest.json").read_text())
    scenarios = {s: read_npz(COMPARISON/city/s/"analysis_matrices.npz") for s in ("with_transit", "walking_only")}
    costs = {s: read_npz(COMPARISON/city/s/"cost_matrices.npz") for s in scenarios}
    scores = {s: pd.read_csv(COMPARISON/city/s/"segregation_comparison.csv").set_index("group") for s in scenarios}
    for data in [modes, *scenarios.values(), *costs.values()]:
        if not np.array_equal(data["tract_geoid"].astype(str), nodes.tract_geoid):
            raise ValueError("Input tract orders differ")
    on, off = scenarios["with_transit"], scenarios["walking_only"]
    if not np.array_equal(costs["with_transit"]["walk_cost_usd"], costs["walking_only"]["walk_cost_usd"]):
        raise ValueError("Scenarios do not use identical walking inputs")
    savings = costs["walking_only"]["C_usd"]-costs["with_transit"]["C_usd"]
    if (savings < -1e-12).any():
        raise ValueError("Adding transit increased generalized cost")
    n = len(nodes)
    nonlocal_mask = ~np.eye(n, dtype=bool)
    no_access = modes["access_stop_count"] == 0
    # These zero-access rows necessarily cannot gain a transit candidate; the
    # unchanged principal submatrix supplies a spectral-radius lower bound.
    if not np.array_equal(on["P"][no_access], off["P"][no_access]):
        raise ValueError("A zero-access tract unexpectedly changes interaction weights")
    access = access_diagnostics(city, nodes, modes, travel)
    changed = savings > 1e-12
    row_variation = .5*np.abs(on["P"]-off["P"]).sum(axis=1)
    city_row = {"city_id": city, "tracts": n, "population": int(nodes.population.sum()),
        "service_dates": ";".join(sorted({f["service_date"] for f in travel["feeds"]})),
        "active_bus_routes": sum(f["used_routes_by_mode"].get("bus", 0) for f in travel["feeds"]),
        "active_rail_routes": sum(f["used_routes_by_mode"].get("rail", 0) for f in travel["feeds"]),
        "nonlocal_pairs": int(nonlocal_mask.sum()),
        "transit_candidate_pairs": int((np.isfinite(modes["transit_seconds"]) & nonlocal_mask).sum()),
        "transit_lowers_cost_pairs": int(changed.sum()),
        "transit_lowers_cost_pair_percent": float(100*changed.sum()/nonlocal_mask.sum()),
        "tracts_with_any_cost_improvement": int(changed.any(axis=1).sum()),
        "population_in_tracts_with_any_cost_improvement_percent": float(100*nodes.population[changed.any(axis=1)].sum()/nodes.population.sum()),
        "tracts_without_modeled_access": int(no_access.sum()),
        "population_without_modeled_access_percent": float(100*nodes.population[no_access].sum()/nodes.population.sum()),
        "population_weighted_transition_total_variation": float(np.average(row_variation, weights=nodes.population)),
        "same_walking_inputs_verified": True,
        "zero_access_P_rows_identical_verified": True}
    for reason in ("no_boardable_stop_within_modeled_radius", "all_nearby_boardable_stops_rejected_by_water_chord_check"):
        city_row[reason+"_tract_count"] = int(access.modeled_access_status.eq(reason).sum())
    category_rows, top_rows, fryer_rows = [], [], []
    for r, category in enumerate(on["group_names"]):
        row = {"city_id": city, "group": str(category), "population_share_K": float(on["counts"][:, r].sum()/on["population"].sum())}
        for metric in ("segregation_index", "segregation_index_normalized", "fryer_majority_ssi", "fryer_majority_ssi_minus_K", "network_isolation", "network_isolation_normalized"):
            for scenario in scenarios:
                row[metric+"_"+scenario] = float(scores[scenario].loc[category, metric])
            row[metric+"_walking_minus_transit"] = row[metric+"_walking_only"]-row[metric+"_with_transit"]
        for scenario, data in scenarios.items():
            A = data["A"][r]
            value, vector = float(data["eigenvalues"][r]), data["right_eigenvectors"][:, r]
            _, elasticity, left_residual = spectral_details(A, value, vector)
            unserved_value = perron_pair(A[np.ix_(no_access, no_access)])[0] if no_access.any() else 0.
            if unserved_value > value+1e-9:
                raise ValueError("Principal submatrix violates Perron monotonicity")
            sorted_indices = np.argsort(elasticity)[::-1]
            row.update({
                "unserved_principal_eigenvalue_"+scenario: unserved_value,
                "unserved_eigenvalue_fraction_of_city_"+scenario: unserved_value/value,
                "right_eigenvector_effective_tracts_"+scenario: float(1/np.square(vector).sum()),
                "row_elasticity_effective_tracts_"+scenario: float(1/np.square(elasticity).sum()),
                "row_elasticity_mass_without_access_"+scenario: float(elasticity[no_access].sum()),
                "right_eigenvector_mass_without_access_"+scenario: float(vector[no_access].sum()),
                "maximum_diagonal_A_"+scenario: float(np.diag(A).max()),
                "maximum_diagonal_fraction_of_eigenvalue_"+scenario: float(np.diag(A).max()/value),
                "top1_row_elasticity_"+scenario: float(elasticity[sorted_indices[:1]].sum()),
                "top5_row_elasticity_"+scenario: float(elasticity[sorted_indices[:5]].sum()),
                "top10_row_elasticity_"+scenario: float(elasticity[sorted_indices[:10]].sum()),
                "left_eigenpair_residual_"+scenario: left_residual,
            })
            for rank, i in enumerate(sorted_indices[:top_count], 1):
                top_rows.append({"group": str(category), "scenario": scenario, "row_elasticity_rank": rank,
                    **access.iloc[i].to_dict(), "city_eigenvalue": value,
                    "group_proportion": float(data["proportions"][i, r]),
                    "right_eigenvector_component": float(vector[i]), "row_elasticity": float(elasticity[i]),
                    "self_interaction_P_ii": float(data["P"][i, i]), "self_race_weight_A_ii": float(A[i, i]),
                    "P_row_total_variation_between_scenarios": float(row_variation[i]),
                    "outgoing_pairs_with_transit_cost_improvement": int(changed[i].sum())})
        selected = np.flatnonzero(on["proportions"][:, r] >= .5)
        row["fryer_majority_tract_count"] = len(selected)
        row["fryer_population_in_majority_tracts"] = int(nodes.population.iloc[selected].sum())
        if len(selected):
            for scenario, data in scenarios.items():
                G = data["P"][np.ix_(selected, selected)]
                value, vector, _, _ = perron_pair(G)
                # All saved complete-comparison matrices have positive support,
                # hence exactly one majority component; validate this premise.
                if not (G > 0).all():
                    raise ValueError("Fryer diagnostics expect one positive majority component")
                if abs(value-float(scores[scenario].loc[category, "fryer_majority_ssi"])) > 1e-9:
                    raise ValueError("Fryer diagnostic eigenvalue differs from saved score")
                _, elasticity, _ = spectral_details(G, value, vector)
                unserved_selected = no_access[selected]
                bound = perron_pair(G[np.ix_(unserved_selected, unserved_selected)])[0] if unserved_selected.any() else 0.
                row["fryer_unserved_principal_eigenvalue_"+scenario] = bound
                row["fryer_row_elasticity_without_access_"+scenario] = float(elasticity[unserved_selected].sum())
                for rank, local_i in enumerate(np.argsort(elasticity)[::-1][:top_count], 1):
                    i = selected[local_i]
                    fryer_rows.append({"group": str(category), "scenario": scenario, "row_elasticity_rank": rank,
                        **access.iloc[i].to_dict(), "fryer_city_ssi": value,
                        "group_proportion": float(data["proportions"][i, r]),
                        "fryer_component_eigenvector": float(vector[local_i]),
                        "fryer_row_elasticity": float(elasticity[local_i]),
                        "self_interaction_P_ii": float(data["P"][i, i])})
        category_rows.append(row)
    return city_row, category_rows, access, top_rows, fryer_rows


def write_findings(cities=("houston", "phoenix"), top_count=10, output=OUTPUT):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    city_rows, categories, accesses, pssi_top, fryer_top = [], [], [], [], []
    for city in cities:
        print(f"Diagnosing saved scenarios for {city}; no routing or network requests", flush=True)
        summary, rows, access, top, fryer = diagnose_city(city, top_count)
        city_rows.append(summary); categories.extend(rows); accesses.append(access); pssi_top.extend(top); fryer_top.extend(fryer)
    frames = {"city_coverage": pd.DataFrame(city_rows), "category_diagnostics": pd.DataFrame(categories),
              "tract_access": pd.concat(accesses, ignore_index=True), "pssi_top_tracts": pd.DataFrame(pssi_top),
              "fryer_top_tracts": pd.DataFrame(fryer_top)}
    for name, frame in frames.items():
        frame.to_csv(output/(name+".csv"), index=False)
    definition = {
        "row_elasticity": "h_i=l_i*v_i/(l^T*v), using positive left/right Perron eigenvectors. h_i is the fractional eigenvalue sensitivity to proportionally scaling row i; it sums to1. This is not population share or the user's tract eigenvector score.",
        "effective_tracts": "Inverse sum of squared normalized components, 1/sum(w_i^2). A value near1 indicates strong concentration.",
        "unserved_principal_eigenvalue": "Spectral radius of the principal A submatrix restricted to tracts with zero modeled boarding access. This submatrix is exactly unchanged between scenarios and provides a lower bound on each complete city eigenvalue. It is a diagnostic, not a recomputed city score with people removed.",
        "self_interaction_P_ii": "Modeled origin-i probability assigned to destination-i; not an observed percentage of within-tract journeys.",
        "self_race_weight_A_ii": "P_ii*p_i; the city PSSI is at least this diagonal entry for a nonnegative A.",
        "modeled_access_status": "Explains exclusion by this model: no active boardable stop within bus1km/rail3km approximate walking radius, or every nearby stop rejected by the straight-chord water test. Neither proves absence of usable real-world transit.",
        "nearest_active_stop_distances": "Active unrestricted boardable GTFS stops on the saved service date, with departure support in07:00–10:00. Geodesic kilometers and kilometers multiplied by1.3 are both given. The saved limits apply to the latter:1km bus and3km rail.",
        "P_row_total_variation_between_scenarios": "0.5*sum_j|P_with_transit[i,j]-P_walking_only[i,j]|. Zero means identical modeled interaction distributions from this tract.",
        "network_isolation": "Mean same-category modeled exposure experienced by category members, sum_i (D_i*p_i/N_r)*(P*p)_i. It measures population-average exposure rather than the strongest spectral cluster.",
        "Fryer": "Existing majority-tract adaptation. A no-majority group has zero SSI by notebook convention, unrelated to actual access. A sole majority tract makes Fryer SSI equal that tract's P_ii.",
    }
    source_paths = [Path(__file__), ROOT/"scripts/multicity_analysis.py", ROOT/"scripts/multicity_transit.py", COMPARISON/"manifest.json"]
    for city in cities:
        source_paths.extend([BASE/city/"travel/modes.npz", BASE/city/"travel/manifest.json", BASE/city/"demographics/nodes.csv"])
        for scenario in ("with_transit", "walking_only"):
            source_paths.extend([COMPARISON/city/scenario/"analysis_matrices.npz", COMPARISON/city/scenario/"cost_matrices.npz"])
    findings = {"cities": city_rows, "categories": categories, "definitions": definition,
        "scope": "Read-only analysis of saved scenarios; static GTFS used only to reproduce access-stop screening, not to reroute.",
        "routes_api_calls": 0, "network_requests": 0, "journeys_recomputed": 0,
        "limitations": ["Zero modeled transit access does not establish no actual transit access. Longer walks, bridges/street detours, different departures and missing agencies can change access.",
            "The common walking matrix is an approximate distance matrix, including unverified crossings; results are model diagnostics, not observations.",
            "Official GTFS availability does not validate the expected-time routing approximation or the chosen fare/time-sensitivity assumptions."],
        "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}}
    (output/"findings.json").write_text(json.dumps(findings, indent=2, allow_nan=False)+"\n")
    lines = ["Why transit-on/off PSSI can be similar: saved-model diagnostics", "", "The similarities are category-specific. Adding transit changes many costs, but a dominant spectral cluster whose model has no boarding access can remain unchanged.", ""]
    for city in city_rows:
        lines.append(f"{city['city_id']}: {city['active_bus_routes']} bus and {city['active_rail_routes']} rail routes were active; transit lowers {city['transit_lowers_cost_pair_percent']:.2f}% of nonlocal pair costs. {city['tracts_without_modeled_access']} of {city['tracts']} tracts have zero modeled boarding access.")
        lines.append(f"Access exclusions: {city['no_boardable_stop_within_modeled_radius_tract_count']} have no active boardable stop inside the configured distance radius; {city['all_nearby_boardable_stops_rejected_by_water_chord_check_tract_count']} have nearby candidates but all fail the conservative water-chord screen.")
        for category in categories:
            if category["city_id"] != city["city_id"]: continue
            lines.append(f"  {category['group']}: normalized PSSI transit={category['segregation_index_normalized_with_transit']:.9f}, walking={category['segregation_index_normalized_walking_only']:.9f}; unchanged zero-access principal block gives {100*category['unserved_eigenvalue_fraction_of_city_with_transit']:.6f}% of transit-on PSSI. Network isolation transit={category['network_isolation_with_transit']:.6f}, walking={category['network_isolation_walking_only']:.6f}.")
        lines.append("")
    lines += ["Interpretation of diagnostics:"]+[f"{key}: {value}" for key, value in definition.items()]
    lines += ["", "No Routes API calls, downloads, or new journey computations were performed."]
    (output/"findings.txt").write_text("\n".join(lines)+"\n")
    write_html_report(output)
    return findings


def write_html_report(output=OUTPUT):
    """Render the saved diagnostics only: no feed parsing, routing, or downloads."""
    output = Path(output)
    cities = pd.read_csv(output/"city_coverage.csv")
    categories = pd.read_csv(output/"category_diagnostics.csv")
    top = pd.read_csv(output/"pssi_top_tracts.csv", dtype={"tract_geoid": str})
    labels = {"houston": "Houston", "phoenix": "Phoenix", "white_alone": "White alone",
              "black_alone": "Black alone", "hispanic_latino_any_race": "Hispanic, any race", "asian_alone": "Asian alone"}

    def labeled(frame):
        return frame.replace({"city_id": labels, "group": labels})

    def table(frame, *, formats=None):
        return '<div class="table-wrap" tabindex="0">' + frame.to_html(index=False, border=0, escape=True,
            float_format=lambda x: f"{x:.6f}", formatters=formats or {}, classes="data-table") + '</div>'

    comparison = labeled(categories[["city_id", "group", "segregation_index_with_transit", "segregation_index_walking_only",
        "segregation_index_normalized_with_transit", "segregation_index_normalized_walking_only",
        "segregation_index_normalized_walking_minus_transit"]]).rename(columns={
            "city_id": "City", "group": "Category", "segregation_index_with_transit": "PSSI: transit",
            "segregation_index_walking_only": "PSSI: walking only", "segregation_index_normalized_with_transit": "Normalized: transit",
            "segregation_index_normalized_walking_only": "Normalized: walking only",
            "segregation_index_normalized_walking_minus_transit": "Normalized change (walking − transit)"})
    coverage = labeled(cities[["city_id", "active_bus_routes", "active_rail_routes", "tracts", "transit_lowers_cost_pair_percent",
        "population_in_tracts_with_any_cost_improvement_percent", "tracts_without_modeled_access",
        "no_boardable_stop_within_modeled_radius_tract_count", "all_nearby_boardable_stops_rejected_by_water_chord_check_tract_count"]]).rename(columns={
            "city_id": "City", "active_bus_routes": "Active bus routes", "active_rail_routes": "Active rail routes", "tracts": "Tract portions",
            "transit_lowers_cost_pair_percent": "Pairs with lower cost (%)", "population_in_tracts_with_any_cost_improvement_percent": "Population at origins with an improvement (%)",
            "tracts_without_modeled_access": "No modeled boarding access", "no_boardable_stop_within_modeled_radius_tract_count": "No stop inside radius",
            "all_nearby_boardable_stops_rejected_by_water_chord_check_tract_count": "All nearby stops rejected by water screen"})
    stable = categories[categories.group.isin(["white_alone", "asian_alone"])].copy()
    stable["bound_percent"] = 100*stable.unserved_eigenvalue_fraction_of_city_with_transit
    bounds = labeled(stable[["city_id", "group", "unserved_principal_eigenvalue_with_transit", "segregation_index_with_transit", "bound_percent"]]).rename(columns={
        "city_id": "City", "group": "Category", "unserved_principal_eigenvalue_with_transit": "Unchanged block eigenvalue",
        "segregation_index_with_transit": "Whole-city PSSI with transit", "bound_percent": "Block / city eigenvalue (%)"})
    drivers = top[(top.scenario == "with_transit") & top.group.isin(["white_alone", "asian_alone"]) & (top.row_elasticity_rank == 1)]
    drivers = labeled(drivers[["city_id", "group", "tract_geoid", "population", "nearest_active_bus_stop_name",
        "nearest_active_bus_stop_walk_proxy_km", "self_interaction_P_ii"]]).rename(columns={
            "city_id": "City", "group": "Category", "tract_geoid": "Example tract GEOID", "population": "Residents in city portion",
            "nearest_active_bus_stop_name": "Nearest active bus stop", "nearest_active_bus_stop_walk_proxy_km": "Modeled walk to bus (km)",
            "self_interaction_P_ii": "Self-interaction Pᵢᵢ"})
    exposure = labeled(stable[["city_id", "group", "network_isolation_with_transit", "network_isolation_walking_only"]]).rename(columns={
        "city_id": "City", "group": "Category", "network_isolation_with_transit": "Population-average isolation: transit",
        "network_isolation_walking_only": "Population-average isolation: walking only"})
    summary_cards = []
    for row in cities.itertuples():
        summary_cards.append(f'<div class="card"><span>{escape(labels.get(row.city_id,row.city_id))}</span>'
            f'<strong>{row.transit_lowers_cost_pair_percent:.1f}%</strong><p>of ordered tract pairs have lower generalized cost with transit.</p>'
            f'<small>Loaded active regional feed: {row.active_bus_routes} bus routes · {row.active_rail_routes} rail routes</small></div>')
    html = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Why Houston and Phoenix PSSI can barely change with transit</title><style>
:root{color-scheme:light}*{box-sizing:border-box}body{margin:0;background:#f3f6fa;color:#172b40;font:16px/1.6 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}main{max-width:1240px;margin:auto;padding:36px 24px 70px}h1{font-size:clamp(27px,3.4vw,40px);line-height:1.15;max-width:1000px;margin:14px 0 22px}h2{font-size:24px;line-height:1.3;margin:0 0 16px}h3{font-size:18px;margin-bottom:6px}p{max-width:1050px;margin:12px 0}a{color:#075e9b;text-underline-offset:3px}.eyebrow{color:#4b6379;font-size:13px;font-weight:700;letter-spacing:.08em;text-transform:uppercase}.lead{font-size:19px;max-width:1060px}.cards{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px;margin:26px 0}.card,section{background:white;border:1px solid #d9e2ed;border-radius:12px;padding:24px}.card span{font-weight:700}.card strong{display:block;font-size:39px;color:#09615f;line-height:1.3}.card p{margin:7px 0}.card small{color:#496275}section{margin:22px 0}.note{padding:18px 20px;border-left:4px solid #c18516;background:#fff4d7;border-radius:4px;margin:18px 0}.formula{padding:15px 18px;border-radius:7px;background:#edf3f8;font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:14px;overflow-wrap:anywhere}.table-wrap{overflow:auto;margin:17px 0;border:1px solid #d9e2ed;border-radius:7px;max-height:590px}.data-table{border-collapse:collapse;width:100%;font-size:14px;line-height:1.45}.data-table th,.data-table td{padding:12px 14px;border-bottom:1px solid #e0e7ef;text-align:left;vertical-align:top}.data-table th{background:#eaf1f8;position:sticky;top:0;min-width:105px}.data-table td{font-variant-numeric:tabular-nums}.data-table tbody tr:nth-child(even){background:#f8fafc}.caption{font-size:14px;color:#466075}li{margin:7px 0}.links{display:flex;flex-wrap:wrap;gap:8px 18px}.footer{font-size:14px;color:#496275}@media(max-width:650px){main{padding:24px 14px}.cards{grid-template-columns:1fr}.card,section{padding:19px}h2{font-size:21px}}@media print{body{background:white}main{padding:0}.table-wrap{overflow:visible;max-height:none}.data-table{font-size:9px}.data-table th{position:static}.data-table th,.data-table td{padding:6px}.cards,section{break-inside:avoid}}
</style></head><body><main>
<div class="eyebrow">Saved-model diagnostic · Houston and Phoenix</div>
<h1>Transit changes many journeys, but a dominant spectral cluster can stay unchanged.</h1>
<p class="lead">The near-equality is mainly in the White and Asian PSSI results. Black and Hispanic PSSI change substantially. Official bus and rail schedule data are already available and used here; near-equal scores do not show that transit data are impossible to obtain.</p>
<div class="cards">''' + ''.join(summary_cards) + '''</div>
<div class="note"><strong>“No modeled boarding access” does not mean “no actual transit.”</strong> It means the representative point could not reach an active, boardable stop under this model’s distance limits and pedestrian-water screen. The diagnostic evaluates an approximation, not observed itineraries.</div>
<section><h2>The score changes depend on the category</h2>
<p>Both scenarios use exactly the same complete distance-based walking matrix and 2020 city-limit populations. Only the saved rail/bus option changes. Normalized PSSI is <strong>(λ − K) / (1 − K)</strong>, where K is the category’s citywide population fraction.</p>'''
    html += table(comparison, formats={"Normalized change (walking − transit)": lambda x: f"{x:.9f}"})
    html += '''<p class="caption">A positive change means the walking-only score is higher. Extra decimals preserve Houston’s small White/Asian differences. Hispanic includes any race and overlaps the race-alone categories.</p>
<p><a href="category_diagnostics.csv">Complete category diagnostics CSV</a> · <a href="../transit_comparison/index.html">Original controlled transit-versus-walking comparison</a></p></section>
<section><h2>Why a citywide eigenvalue can barely move</h2>
<p>PSSI is the largest eigenvalue of <strong>A = P diag(p)</strong>. It can be controlled by a strongly connected group of high-same-race tracts, even when transit changes interactions elsewhere. It is not an average of all travel improvements.</p>
<p>For tracts with zero modeled boarding access, the entire row of P is exactly identical in the two scenarios. The principal A block formed from those tracts is therefore unchanged. Its eigenvalue supplies a lower bound on the whole-city PSSI:</p>
<div class="formula">ρ(A restricted to zero-access tracts) ≤ ρ(A) = city PSSI</div>'''
    html += table(bounds)
    html += '''<p class="caption">“Block / city eigenvalue” measures how tightly this unchanged block bounds the transit-on eigenvalue. It is not a population percentage or a share of trips. No people were removed to recompute a different city score.</p>
<p>For Houston White and Asian, that bound is almost the full city score. Phoenix White is also tightly bounded. Transit still changes many other rows; the dominant cluster largely determines the reported maximum eigenvalue.</p></section>
<section><h2>What excluded boarding access?</h2>
<p>The model permits a walk of at most <strong>1 km to bus</strong> or <strong>3 km to rail</strong>, after multiplying great-circle distance by <strong>1.3</strong>. These correspond to approximately 0.77 km and 2.31 km in straight-line distance. Nearby candidate legs are also rejected if their straight chord intersects mapped water.</p>'''
    html += table(coverage, formats={"Pairs with lower cost (%)": lambda x: f"{x:.2f}", "Population at origins with an improvement (%)": lambda x: f"{x:.2f}"})
    html += '''<p>Almost all zero-access tracts fail the configured radius: 296 of 297 in Houston, and 61 of 62 in Phoenix. Only one tract in each city has nearby candidates that are all rejected by the water screen. This reproduces the saved access counts exactly.</p>
<p class="caption">Active route counts describe the regional feeds used by the model, not a count of routes entirely within city limits. “Lower cost” combines travel time and the illustrative fare; it is not an observed ridership benefit. Both feeds use October 20, 2026 service in the 07:00–10:00 analysis window.</p>
<h3>Examples that strongly influence the spectral score</h3>'''
    html += table(drivers, formats={"Modeled walk to bus (km)": lambda x: f"{x:.3f}", "Self-interaction Pᵢᵢ": lambda x: f"{x:.4f}", "Residents in city portion": lambda x: f"{x:,}"})
    html += '''<p>Houston’s influential White tracts are near Kingwood Park &amp; Ride; its influential Asian tract portions are near Cypress Park &amp; Ride. Some are just beyond the 1 km bus-access limit. Phoenix’s strongest White driver is much farther from an active stop, so a modest increase in the limit would not address that example.</p>
<p class="caption">Examples are ranked by left/right eigenvector row elasticity, a diagnostic of eigenvalue sensitivity—not by population. Pᵢᵢ is the model’s self-interaction weight, not an observed fraction of residents’ trips. See <a href="pssi_top_tracts.csv">all leading tracts</a> and <a href="tract_access.csv">every tract’s access distances</a>.</p></section>
<section><h2>Population-average exposure still changes</h2>
<p>A complementary measure averages modeled same-category exposure over that category’s residents. This network-isolation measure falls with transit for all four examples below, even where PSSI barely moves.</p>'''
    html += table(exposure)
    html += '''<div class="formula">Network isolation = Σᵢ (Dᵢ pᵢ / Nᵣ) (P p)ᵢ</div>
<p class="caption">This is a different summary of the same model. It does not replace the requested PSSI.</p></section>
<section><h2>What this says about data accuracy</h2>
<p><strong>The data are obtainable; the current travel model is approximate.</strong> The saved feeds contain stops, routes and schedules for the relevant bus and rail systems. More accurate access paths would require a pedestrian street/bridge network; journey estimates could use exact scheduled departures and a wider set of times. Sensitivity checks on access limits, representative points, fares and cost sensitivity would show how robust the scores are.</p>
<p>These results use expected waiting and median ride times, a standardized $3 transit fare, $15/hour value of time, β = 0.1 per dollar, and a $1.25 local cost floor. Walking is distance × 1.3 at 4.8 km/h, and the shared complete walking matrix can include unverified crossings. Better data and routing can improve the model; these diagnostics do not establish that the current travel times are accurate.</p>
<div class="note"><strong>Fryer note:</strong> the Fryer fields in this diagnostic’s archived CSVs refer to the previous majority-tract adaptation. They are historical and should not be substituted for the current <a href="../../distance_fryer/index.html">distance-based Fryer analysis</a>. In that old adaptation, Phoenix Black had no majority tracts, while Phoenix Asian’s only majority tract portion contained one resident—important artifacts when interpreting those earlier unchanged Fryer values.</div></section>
<section><h2>Files and reproducibility</h2><div class="links">
<a href="city_coverage.csv">City coverage CSV</a><a href="category_diagnostics.csv">Category diagnostics CSV</a><a href="pssi_top_tracts.csv">Leading PSSI tracts CSV</a><a href="tract_access.csv">All access checks CSV</a><a href="findings.json">Definitions and checksums JSON</a><a href="findings.txt">Plain-text findings</a><a href="../transit_sources.json">Transit source manifest</a></div>
<p>Generated from saved outputs by <code>scripts/diagnose_transit_similarity.py --report-only</code>. The full diagnostic checks identical walking inputs, reproduces saved stop-access counts, verifies left/right eigenvalues, and checks the unchanged-block eigenvalue bounds. This report reads those results; it does not reroute.</p></section>
<p class="footer">No Routes API calls, downloads, or new journey computations were performed for these diagnostics.</p>
</main></body></html>'''
    (output/"index.html").write_text(html)
    return output/"index.html"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cities", nargs="+", default=["houston", "phoenix"])
    parser.add_argument("--top-count", type=int, default=10)
    parser.add_argument("--report-only", action="store_true", help="Render index.html from existing diagnostics without recomputation")
    args = parser.parse_args()
    if args.report_only:
        write_html_report()
    else:
        write_findings(args.cities, args.top_count)
