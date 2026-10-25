"""Exact 2020 city-limit Census counts aggregated from blocks to tract portions.

Only --download makes network requests, exclusively to public census.gov files.
No Google Routes API is used. Race categories overlap with Hispanic ethnicity.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time
import urllib.request
from zipfile import ZipFile

import numpy as np
import pandas as pd
from pyproj import CRS, Transformer

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/multicity/census"
OUTPUT = ROOT / "outputs/multicity"
LAYOUT = ROOT / "data/supplements/2020PL_R_import_scripts.zip"
LEGACY_BASE = "https://www2.census.gov/programs-surveys/decennial/2020/data/01-Redistricting_File--PL_94-171"
TIGER_BASE = "https://www2.census.gov/geo/tiger/TIGER2020PL/LAYER"
TECH_DOC = "https://www2.census.gov/programs-surveys/decennial/2020/technical-documentation/complete-tech-docs/summary-file/2020Census_PL94_171Redistricting_StatesTechDoc_English.pdf"
STATES = {
    "42": ("pa", "Pennsylvania"), "12": ("fl", "Florida"),
    "06": ("ca", "California"), "49": ("ut", "Utah"),
    "11": ("dc", "District_of_Columbia"), "48": ("tx", "Texas"),
    "26": ("mi", "Michigan"), "17": ("il", "Illinois"),
    "04": ("az", "Arizona"), "13": ("ga", "Georgia"),
}
CITIES = {
    "philadelphia": ("Philadelphia", "42", "60000", 1603797),
    "miami": ("Miami", "12", "45000", 442241),
    "los_angeles": ("Los Angeles", "06", "44000", 3898747),
    "salt_lake_city": ("Salt Lake City", "49", "67000", 199723),
    "san_francisco": ("San Francisco", "06", "67000", 873965),
    "washington_dc": ("Washington", "11", "50000", 689545),
    "houston": ("Houston", "48", "35000", 2304580),
    "detroit": ("Detroit", "26", "22000", 639111),
    "chicago": ("Chicago", "17", "14000", 2746388),
    "phoenix": ("Phoenix", "04", "55000", 1608139),
    "atlanta": ("Atlanta", "13", "04000", 498715),
}
VARIABLES = {
    "population": "P0010001", "white_alone": "P0010003",
    "black_alone": "P0010004", "hispanic_latino_any_race": "P0020002",
    "asian_alone": "P0010006",
}
COUNT_NAMES = list(VARIABLES)
RACE_NAMES = COUNT_NAMES[1:]
GEO_FIELDS = ["LOGRECNO", "SUMLEV", "GEOVAR", "GEOCOMP", "CHARITER",
              "GEOCODE", "STATE", "COUNTY", "PLACE", "TRACT", "BLOCK",
              "NAME", "POP100", "INTPTLAT", "INTPTLON", "AREALAND", "AREAWATR"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def state_sources(state, geometry=True):
    abbr, folder = STATES[state]
    result = [(f"{LEGACY_BASE}/{folder}/{abbr}2020.pl.zip", DATA / f"{abbr}2020.pl.zip")]
    if geometry:
        for layer in ("PLACE", "TRACT"):
            name = f"tl_2020_{state}_{layer.lower()}20.zip"
            result.append((f"{TIGER_BASE}/{layer}/2020/{name}", DATA / name))
    return result


def download_file(url, path):
    """Atomic cached downloads, validated as ZIP archives before use."""
    if path.exists():
        with ZipFile(path) as archive:
            if archive.testzip() is None:
                return {"url": url, "path": str(path.relative_to(ROOT)), "cached": True}
    require(url.startswith("https://www2.census.gov/"), "Only Census public downloads are permitted")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".part")
    for attempt in range(4):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "APMA4903 academic census analysis"})
            with urllib.request.urlopen(request, timeout=180) as response, temp.open("wb") as out:
                for chunk in iter(lambda: response.read(4 * 1024 * 1024), b""):
                    out.write(chunk)
            with ZipFile(temp) as z:
                require(z.testzip() is None, f"Corrupt archive {url}")
            temp.replace(path)
            return {"url": url, "path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size}
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def fetch(states, geometry=True):
    jobs = [source for state in sorted(set(states)) for source in state_sources(state, geometry)]
    with ThreadPoolExecutor(max_workers=3) as pool:
        pending = {pool.submit(download_file, url, path): url for url, path in jobs}
        for future in as_completed(pending):
            print(json.dumps({"download": future.result()}), flush=True)


def geo_layout():
    with ZipFile(LAYOUT) as archive:
        layout = archive.read("R Versions/pl_geohd_2020_dar.r").decode()
    return re.findall(r'"([A-Z0-9]+)"', layout.split("colnames(header) <- c(")[1])


def read_state(state, cities):
    """Join P1/P2 to selected block/place/tract geography by LOGRECNO."""
    places = {CITIES[city][2] for city in cities}
    abbr = STATES[state][0]
    selected_geo = []
    with ZipFile(DATA / f"{abbr}2020.pl.zip") as archive:
        with archive.open(f"{abbr}geo2020.pl") as stream:
            for chunk in pd.read_csv(stream, sep="|", header=None, names=geo_layout(),
                    usecols=GEO_FIELDS, dtype=str, keep_default_na=False, chunksize=100000,
                    encoding="latin-1"):
                valid = chunk.GEOCOMP.eq("00") & chunk.GEOVAR.eq("00") & chunk.CHARITER.eq("000")
                keep = chunk.SUMLEV.eq("140") | (chunk.SUMLEV.isin(["160", "750"]) & chunk.PLACE.isin(places))
                selected_geo.append(chunk.loc[valid & keep].copy())
        geo = pd.concat(selected_geo, ignore_index=True)
        ids = set(geo.LOGRECNO)
        names = ["FILEID", "STUSAB", "CHARITER", "CIFSN", "LOGRECNO"]
        names += [f"P001{i:04d}" for i in range(1, 72)]
        names += [f"P002{i:04d}" for i in range(1, 74)]
        variables = list(VARIABLES.values()) + ["P0020001"]
        selected_data = []
        with archive.open(f"{abbr}000012020.pl") as stream:
            for chunk in pd.read_csv(stream, sep="|", header=None, names=names,
                    usecols=["LOGRECNO", *variables], dtype={"LOGRECNO": str}, chunksize=100000):
                selected_data.append(chunk.loc[chunk.LOGRECNO.isin(ids)].copy())
    records = geo.merge(pd.concat(selected_data, ignore_index=True), on="LOGRECNO",
                        how="left", validate="one_to_one")
    require(records[variables].notna().all().all(), "Missing P1/P2 records")
    require((records[variables] >= 0).all().all(), "Negative Census count")
    require(np.array_equal(records.P0010001, records.P0020001), "P1/P2 population disagreement")
    require(np.array_equal(records.P0010001, records.POP100.astype(np.int64)), "Header/table population disagreement")
    for variable in VARIABLES.values():
        require((records[variable] <= records.P0010001).all(), "Category exceeds total population")
    records = records.rename(columns={v: k for k, v in VARIABLES.items()})
    records[COUNT_NAMES] = records[COUNT_NAMES].astype(np.int64)
    return records


def aggregate_city(records, city):
    """Sum only blocks belonging to this 2020 Census place; never area-weight race."""
    name, state, place, expected_population = CITIES[city]
    blocks = records.loc[records.SUMLEV.eq("750") & records.PLACE.eq(place)].copy()
    published = records.loc[records.SUMLEV.eq("160") & records.PLACE.eq(place)]
    require(len(published) == 1, f"Expected one city record: {city}")
    official = published.iloc[0]
    require(official.NAME in (name + " city", name), f"Unexpected place name {official.NAME}: {city}")
    require(int(official.population) == expected_population, f"Unexpected 2020 population: {city}")
    require(blocks.GEOCODE.is_unique and blocks.GEOCODE.str.len().eq(15).all(), "Block GEOIDs must be unique and complete")
    require(np.array_equal(blocks[COUNT_NAMES].sum().to_numpy(), official[COUNT_NAMES].to_numpy()),
            f"Block sum differs from separately published city P1/P2 totals: {city}")
    blocks["tract_geoid"] = blocks.GEOCODE.str[:11]
    grouped = blocks.groupby("tract_geoid", sort=True)[COUNT_NAMES].sum()
    grouped["census_block_count"] = blocks.groupby("tract_geoid").size()
    full_tracts = records.loc[records.SUMLEV.eq("140")].set_index("GEOCODE")
    require(set(grouped.index) <= set(full_tracts.index), "Missing full tract records")
    grouped["full_tract_population"] = full_tracts.loc[grouped.index, "population"]
    grouped["city_share_of_tract_population"] = np.divide(grouped.population,
        grouped.full_tract_population, out=np.zeros(len(grouped), dtype=float), where=grouped.full_tract_population.to_numpy() > 0)
    # Population alone cannot detect an unpopulated tract area outside the city.
    # Geometry metadata separately identifies intersected tract portions.
    require((grouped.population <= grouped.full_tract_population).all(), "City tract count exceeds full tract")
    zero = grouped.loc[grouped.population.eq(0)].reset_index()
    grouped = grouped.loc[grouped.population.gt(0)].copy()
    projection = CRS.from_proj4(f"+proj=laea +lat_0={float(official.INTPTLAT)} +lon_0={float(official.INTPTLON)} +datum=WGS84 +units=m")
    forward = Transformer.from_crs(4326, projection, always_xy=True)
    backward = Transformer.from_crs(projection, 4326, always_xy=True)
    populated = blocks.loc[blocks.population.gt(0)].copy()
    populated["x"], populated["y"] = forward.transform(populated.INTPTLON.astype(float).to_numpy(), populated.INTPTLAT.astype(float).to_numpy())
    representatives = []
    for tract, frame in populated.groupby("tract_geoid", sort=True):
        center = np.average(frame[["x", "y"]], axis=0, weights=frame.population)
        centroid_lon, centroid_lat = backward.transform(*center)
        # An internal populated-block point avoids a centroid falling in water or
        # outside an irregular city-limit tract portion; choose nearest in meters.
        nearest = ((frame.x - center[0]) ** 2 + (frame.y - center[1]) ** 2).idxmin()
        representative = frame.loc[nearest]
        representatives.append({"tract_geoid": tract, "latitude": float(representative.INTPTLAT),
            "longitude": float(representative.INTPTLON), "population_centroid_latitude": centroid_lat,
            "population_centroid_longitude": centroid_lon, "representative_block_geoid": representative.GEOCODE,
            "representative_offset_m": float(np.linalg.norm(representative[["x", "y"]].to_numpy(dtype=float) - center))})
    nodes = grouped.join(pd.DataFrame(representatives).set_index("tract_geoid"), validate="one_to_one").reset_index()
    nodes.insert(0, "matrix_index", np.arange(len(nodes)))
    front = ["matrix_index", "tract_geoid", "latitude", "longitude", *COUNT_NAMES]
    nodes = nodes[front + [column for column in nodes if column not in front]]
    require(np.array_equal(nodes[COUNT_NAMES].sum().to_numpy(), official[COUNT_NAMES].to_numpy()), "Zero-pop exclusion lost residents")
    require(nodes[["latitude", "longitude"]].notna().all().all(), "Missing representative points")
    summary = {"city": city, "city_name": name, "census_place_name": official.NAME,
        "state_fips": state, "place_fips": place, "census_place_geoid": state + place,
        "census_year": 2020, "total_population": int(nodes.population.sum()),
        "analysis_node_count": len(nodes), "all_city_tract_portions": len(grouped) + len(zero),
        "zero_population_portions_excluded": len(zero), "blocks_in_city": len(blocks),
        "populated_blocks_in_city": len(populated),
        "tracts_with_population_outside_city": int((nodes.population < nodes.full_tract_population).sum()),
        "race_counts": {race: int(nodes[race].sum()) for race in RACE_NAMES},
        "race_population_shares": {race: float(nodes[race].sum() / nodes.population.sum()) for race in RACE_NAMES},
        "city_limits": "2020 Census legal/statistical place boundary, not metropolitan area or current municipal boundary",
        "node_definition": "2020 Census tract portion inside the city, summing only SUMLEV750 blocks with the selected PLACE code",
        "representative_point": "Populated Census block internal point nearest the population-weighted centroid of block internal points in a city-centered Lambert azimuthal equal-area projection",
        "population_centroid_columns": "Unsnapped projected population-weighted centroids retained separately for sensitivity analysis",
        "race_definition": "White alone, Black alone and Asian alone include Hispanic persons. Hispanic/Latino is any race; four categories overlap and do not sum to total population.",
        "counts": "Published 2020 P.L. 94-171 P1/P2 values, all ages, April 1 2020; subject to Census disclosure avoidance. No interpolation or population allocation by area.",
        "variables": VARIABLES, "all_validation_checks_passed": True,
        "validation_checks": ["City place name matches expected FIPS", "P1 and P2 populations match header POP100", "Block counts match separately published SUMLEV160 place total for all five columns", "City block GEOIDs unique", "City portions do not exceed full tract population", "Zero-population exclusion preserves all city counts"],
        "google_routes_api_calls": 0}
    return nodes, zero, blocks, summary


def export_geometry(nodes, city, output):
    import geopandas as gpd
    from shapely import make_valid
    name, state, place, _ = CITIES[city]
    places = gpd.read_file(DATA / f"tl_2020_{state}_place20.zip")
    place_field = "PLACEFP20" if "PLACEFP20" in places else "PLACEFP"
    boundary = places.loc[places[place_field].eq(place)].to_crs(4326)
    require(len(boundary) == 1, f"Missing or duplicated place shape {city}")
    boundary.to_file(output / "city_boundary.geojson", driver="GeoJSON")
    tracts = gpd.read_file(DATA / f"tl_2020_{state}_tract20.zip")
    geoid_field = "GEOID20" if "GEOID20" in tracts else "GEOID"
    tracts = tracts.loc[tracts[geoid_field].isin(nodes.tract_geoid)].to_crs(4326)
    tracts["tract_geoid"] = tracts[geoid_field]
    require(set(tracts.tract_geoid) == set(nodes.tract_geoid), "Missing tract geometries")
    boundary_geom = make_valid(boundary.geometry.iloc[0])
    tracts.geometry = tracts.geometry.map(make_valid).intersection(boundary_geom)
    require((~tracts.geometry.is_empty).all(), "City intersection removed a populated tract")
    geometry = nodes.merge(tracts[["tract_geoid", "geometry"]], on="tract_geoid", validate="one_to_one")
    gdf = gpd.GeoDataFrame(geometry, geometry="geometry", crs=4326)
    gdf.to_file(output / "nodes.geojson", driver="GeoJSON")
    points = gpd.points_from_xy(gdf.longitude, gdf.latitude)
    require(all(shape.covers(point) for shape, point in zip(gdf.geometry, points)), "Representative outside city tract portion")
    return {"geometry": "Full-resolution 2020 TIGER/Line tract shapes clipped to 2020 TIGER/Line place boundary", "representative_points_inside_node_geometry": True}


def build(cities=None, download=False, geometry=True):
    cities = list(cities or CITIES)
    states = sorted({CITIES[city][1] for city in cities})
    if download:
        fetch(states, geometry=geometry)
    summaries = []
    for state in states:
        selected = [city for city in cities if CITIES[city][1] == state]
        records = read_state(state, selected)
        for city in selected:
            nodes, zero, blocks, summary = aggregate_city(records, city)
            output = OUTPUT / city / "demographics"
            output.mkdir(parents=True, exist_ok=True)
            nodes.to_csv(output / "nodes.csv", index=False)
            zero.to_csv(output / "zero_population_tract_portions.csv", index=False)
            blocks[["GEOCODE", "tract_geoid", "PLACE", "INTPTLAT", "INTPTLON", *COUNT_NAMES]].to_csv(output / "city_blocks.csv.gz", index=False)
            np.savez_compressed(output / "demographics.npz", tract_geoids=nodes.tract_geoid.to_numpy(dtype=str),
                population=nodes.population.to_numpy(np.int64), group_names=np.array(RACE_NAMES),
                counts=nodes[RACE_NAMES].to_numpy(np.int64), proportions=nodes[RACE_NAMES].to_numpy()/nodes.population.to_numpy()[:,None],
                latitude=nodes.latitude.to_numpy(), longitude=nodes.longitude.to_numpy())
            if geometry:
                summary.update(export_geometry(nodes, city, output))
            summary["sources"] = [{"url": url, "path": str(path.relative_to(ROOT)), "sha256": sha256(path), "bytes": path.stat().st_size}
                for url, path in state_sources(state, geometry)]
            summary["sources"].append({"url": TECH_DOC, "role": "Geographic summary levels, PLACE membership and P1/P2 definitions"})
            summary["layout_file"] = {"path": str(LAYOUT.relative_to(ROOT)), "sha256": sha256(LAYOUT)}
            summary["created_utc"] = datetime.now(timezone.utc).isoformat()
            (output / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
            summaries.append(summary)
            print(json.dumps({"city": city, "nodes": len(nodes), "population": summary["total_population"], "validated": True}), flush=True)
    return summaries


def build_water(cities=None, download=False):
    """Optional county-wide mapped water, for conservative walking-chord flags.

    Tract/place polygons themselves are not land masks. These water polygons
    still do not contain a pedestrian network or bridge-crossing permissions.
    """
    import geopandas as gpd
    from shapely import make_valid
    cities = list(cities or CITIES)
    county_map = {}
    for city in cities:
        blocks = pd.read_csv(OUTPUT / city / "demographics/city_blocks.csv.gz",
                             usecols=["GEOCODE"], dtype=str)
        county_map[city] = sorted(set(blocks.GEOCODE.str[:5]))
    counties = sorted({county for values in county_map.values() for county in values})
    sources = {county: (f"https://www2.census.gov/geo/tiger/TIGER2020/AREAWATER/tl_2020_{county}_areawater.zip",
                       DATA / f"tl_2020_{county}_areawater.zip") for county in counties}
    if download:
        with ThreadPoolExecutor(max_workers=3) as pool:
            pending = {pool.submit(download_file, *source): county for county, source in sources.items()}
            for future in as_completed(pending):
                print(json.dumps({"water_download": future.result()}), flush=True)
    for city in cities:
        frames = []
        for county in county_map[city]:
            water = gpd.read_file(sources[county][1]).to_crs(4326)[["geometry"]]
            water["county_geoid"] = county
            water.geometry = water.geometry.map(make_valid)
            frames.append(water)
        water = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=4326)
        output = OUTPUT / city / "demographics"
        water.to_file(output / "water.geojson", driver="GeoJSON")
        summary = {"city": city, "counties": county_map[city], "water_polygon_count": len(water),
            "geometry": "Full-resolution 2020 TIGER/Line AREAWATER polygons for every county containing a city-owned Census block; county-wide, not city-clipped",
            "limitations": ["Mapped water is not a pedestrian network", "A straight line intersecting water may be traversable by a bridge or avoidable by a street detour", "Areas outside these counties have no water coverage guarantee", "Absence of a water intersection does not establish that a path is walkable"],
            "sources": [{"url": sources[county][0], "path": str(sources[county][1].relative_to(ROOT)),
                "sha256": sha256(sources[county][1])} for county in county_map[city]],
            "google_routes_api_calls": 0}
        (output / "water_manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps({"city": city, "water_polygons": len(water)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cities", nargs="+", choices=list(CITIES))
    parser.add_argument("--download", action="store_true", help="Fetch public Census data (never Routes API)")
    parser.add_argument("--no-geometry", action="store_true")
    parser.add_argument("--water-only", action="store_true", help="Build optional mapped-water masks after Census nodes exist")
    args = parser.parse_args()
    if args.water_only:
        build_water(args.cities, download=args.download)
    else:
        build(args.cities, download=args.download, geometry=not args.no_geometry)
