"""Map revised PSSI eigenvectors from saved local arrays; no network requests.

Every panel uses n*v_i (section mean one) on the SAME log color scale 0.25--4.
Out-of-range values are explicitly saturated, counted, and retained in CSVs.
The original scores, geometry assignments and previously generated maps are
never modified. Water is removed only from cartographic display geometry.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from html import escape
import json
import os
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
REVISION = ROOT / "outputs/multicity/walk_access_revision"
OUTPUT = REVISION / "maps"
os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT / ".matplotlib"))

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.colors import FuncNorm
import numpy as np
import pandas as pd
from pyproj import CRS, Transformer
import shapely

CITIES = {
    "new_york_city": "New York City", "philadelphia": "Philadelphia",
    "miami": "Miami", "los_angeles": "Los Angeles",
    "salt_lake_city": "Salt Lake City", "san_francisco": "San Francisco",
    "washington_dc": "Washington, DC", "houston": "Houston",
    "detroit": "Detroit", "chicago": "Chicago", "phoenix": "Phoenix",
    "atlanta": "Atlanta",
}
CATEGORIES = {
    "white_alone": "White alone", "black_alone": "Black alone",
    "hispanic_latino_any_race": "Hispanic · any race", "asian_alone": "Asian alone",
}
NYC_LABELS = {"Bronx": (-73.862, 40.866), "Manhattan": (-74.033, 40.799),
              "Queens": (-73.802, 40.734), "Brooklyn": (-73.950, 40.631),
              "Staten Island": (-74.164, 40.574)}
INK, WATER = "#172b3a", "#edf3f7"
VMIN, VMAX = .25, 4.


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def polygonal_only(shape):
    """Remove zero-area line artifacts left by Census clipping operations."""
    if shape.geom_type == "Polygon":
        return shape
    parts = []
    for part in getattr(shape, "geoms", []):
        if part.geom_type == "Polygon":
            parts.append(part)
        elif hasattr(part, "geoms"):
            parts.append(polygonal_only(part))
    return shapely.union_all(parts)


def shared_norm():
    # Zero numerical components are deliberately assigned the below-range color,
    # not removed by LogNorm's usual masking. Exact values remain in the CSV.
    return FuncNorm((lambda x: np.log(np.maximum(x, np.finfo(float).tiny)), np.exp),
                    vmin=VMIN, vmax=VMAX, clip=False)


def load_city(city_id):
    folder = REVISION / city_id
    arrays_path = folder / "with_transit/analysis_matrices.npz"
    nodes_path = folder / "nodes.csv"
    with np.load(arrays_path, allow_pickle=False) as saved:
        arrays = {name: saved[name] for name in saved.files}
    nodes = pd.read_csv(nodes_path, dtype={"tract_geoid": str, "zone_id": str})
    require("tract_geoid" in nodes, "Revised nodes must expose tract_geoid (NYC uses zone IDs)")
    require(nodes.matrix_index.tolist() == list(range(len(nodes))), "Node matrix_index ordering differs")
    require(nodes.tract_geoid.is_unique, "Duplicate revised node IDs")
    require(np.array_equal(arrays["tract_geoid"], nodes.tract_geoid), "Eigenvector/node ID ordering differs")
    require(arrays["group_names"].tolist() == list(CATEGORIES), "Unexpected category order")
    require(np.array_equal(arrays["population"], nodes.population), "Population alignment differs")
    require(np.array_equal(arrays["counts"], nodes[list(CATEGORIES)].to_numpy()), "Category counts differ from saved nodes")
    n = len(nodes)
    P, A, p, v, lambdas = (arrays[key] for key in ("P", "A", "proportions", "right_eigenvectors", "eigenvalues"))
    require(P.shape == (n, n) and A.shape == (4, n, n) and v.shape == (n, 4), "Unexpected matrix dimensions")
    require(np.isfinite(v).all() and (v >= 0).all(), "Negative/nonfinite right eigenvector")
    require(np.isfinite(P).all() and (P >= 0).all(), "Invalid population transition matrix")
    require(np.isfinite(A).all() and (A >= 0).all(), "Invalid A matrices")
    require(np.allclose(P.sum(axis=1), 1, atol=1e-11, rtol=0), "P rows do not sum to one")
    require(np.allclose(v.sum(axis=0), 1, atol=1e-11, rtol=0), "Right vectors do not sum to one")
    require(np.allclose(arrays["counts"]/arrays["population"][:, None], p, atol=1e-14, rtol=0), "Counts/proportions differ")
    residuals = []
    for k, category in enumerate(CATEGORIES):
        require(np.allclose(A[k], P*p[:, k][None, :], atol=1e-13, rtol=0), f"A definition differs for {category}")
        residual = float(np.max(np.abs(A[k] @ v[:, k] - lambdas[k]*v[:, k])))
        require(residual < 1e-9, f"Eigenpair residual too large for {category}: {residual}")
        residuals.append(residual)
    manifests = {}
    sources = [nodes_path, arrays_path]
    for name, path in (("travel", folder/"travel/manifest.json"), ("analysis", folder/"with_transit/manifest.json")):
        if path.exists():
            manifests[name] = json.loads(path.read_text())
            sources.append(path)
    return nodes, arrays, residuals, manifests, sources


def display_geometry(city_id, nodes):
    """Join exact IDs, project locally, and apply saved water data to display only."""
    prepared=ROOT / "reproducibility/map_geometry" / f"{city_id}.json.gz"
    if prepared.exists():
        import gzip
        with gzip.open(prepared,"rt") as stream: data=json.load(stream)
        require(data["node_ids"] == nodes.tract_geoid.tolist(), "Published geometry node IDs differ")
        projection=CRS.from_wkt(data["projection_wkt"])
        frame=gpd.GeoDataFrame({"tract_geoid":data["node_ids"]}, geometry=[shapely.from_wkb(bytes.fromhex(w)) for w in data["geometry_wkb_hex"]], crs=projection)
        if data["borough"] is not None:frame["borough"]=data["borough"]
        absent=np.asarray(data["absent"],dtype=bool)
        points=gpd.GeoSeries(gpd.points_from_xy(nodes.longitude,nodes.latitude),crs=4326).to_crs(projection)
        outline=frame.dissolve(by="borough").geometry if city_id=="new_york_city" else gpd.GeoSeries([frame.geometry.union_all()],crs=projection)
        return frame,points,outline,absent,projection,[prepared],data["display_method"]

    nyc = city_id == "new_york_city"
    source = ROOT / ("outputs/nyc_routes/zones.geojson" if nyc else f"outputs/multicity/{city_id}/demographics/nodes.geojson")
    frame = gpd.read_file(source)
    key = "zone_id" if nyc else "tract_geoid"
    require(frame[key].is_unique and set(frame[key]) == set(nodes.tract_geoid), "Map geometry IDs differ from revised nodes")
    frame = frame.set_index(key).loc[nodes.tract_geoid].reset_index().rename(columns={key: "section_id"})
    population_field = "TOTPOP" if nyc else "population"
    require(np.array_equal(frame[population_field], nodes.population), "Geometry population alignment differs")
    require(np.array_equal(frame.matrix_index, nodes.matrix_index), "Geometry matrix_index ordering differs")
    projection = CRS.from_proj4(f"+proj=laea +lat_0={nodes.latitude.mean()} +lon_0={nodes.longitude.mean()} +datum=WGS84 +units=m")
    frame = frame.to_crs(projection)
    frame.geometry = frame.geometry.map(shapely.make_valid).map(polygonal_only)
    sources = [source]
    if nyc:
        blockpath = ROOT / "data/raw/tl_2020_36_tabblock20.shp"
        blocks = gpd.read_file(blockpath, where="COUNTYFP20 IN ('005','047','061','081','085')",
            columns=["COUNTYFP20", "ALAND20", "geometry"]).to_crs(projection)
        require(len(blocks) > 0, "NYC land-block query is empty")
        land = blocks.loc[pd.to_numeric(blocks.ALAND20) > 0].geometry.union_all()
        frame.geometry = frame.geometry.intersection(land).map(polygonal_only)
        sources += [blockpath, blockpath.with_suffix(".dbf")]
        method = "Saved NYC sections intersected with union of saved Census blocks with positive land area; no statistical assignments changed."
    else:
        waterpath = ROOT / f"outputs/multicity/{city_id}/demographics/water.geojson"
        water = gpd.read_file(waterpath).to_crs(projection)
        # County-wide water is larger than the city; filter by the city bounding
        # box before union for efficiency, without altering intersecting water.
        left, bottom, right, top = frame.total_bounds
        relevant = water.loc[water.geometry.intersects(shapely.box(left, bottom, right, top))]
        mask = relevant.geometry.map(shapely.make_valid).union_all()
        frame.geometry = frame.geometry.difference(mask).map(polygonal_only)
        sources += [waterpath]
        method = "Saved city-limit tract portions minus saved Census mapped-water polygons for display only; statistical assignments unchanged."
    require(frame.geometry.is_valid.all(), "Invalid display geometry after water removal")
    absent = frame.geometry.is_empty.to_numpy()
    points = gpd.GeoSeries(gpd.points_from_xy(nodes.longitude, nodes.latitude), crs=4326).to_crs(projection)
    outline = (frame.dissolve(by="borough").geometry if nyc else gpd.GeoSeries([frame.geometry.union_all()], crs=projection))
    return frame, points, outline, absent, projection, sources, method


def range_text(values):
    number = lambda value: f"{value:.4g}"
    text = f"Observed range {number(values.min())}–{number(values.max())}  |  below 0.25: {(values < VMIN).sum()}  |  above 4: {(values > VMAX).sum()}"
    if (values == 0).any():
        text += f"  |  zeros: {(values == 0).sum()}"
    return text


def draw_map(ax, city_id, frame, points, outline, absent, projection, values, *, small=False):
    norm = shared_norm()
    cmap = matplotlib.colormaps["RdBu_r"].copy()
    cmap.set_under("#082e50"); cmap.set_over("#67001f")
    ax.set_facecolor(WATER)
    plotframe = frame.loc[~absent]
    plotframe.plot(ax=ax, column=values[~absent], cmap=cmap, norm=norm,
                   edgecolor="white", linewidth=.22 if small else .32, antialiased=True)
    outline.boundary.plot(ax=ax, color="#49616f", linewidth=.55 if small else .7, alpha=.85)
    if absent.any():
        ax.scatter(points[absent].x, points[absent].y, c=values[absent], cmap=cmap, norm=norm,
                   marker="D", s=20 if small else 30, edgecolors=INK, linewidths=.4, zorder=10)
    left, bottom, right, top = frame.total_bounds
    if absent.any():
        point_left, point_bottom, point_right, point_top = points[absent].total_bounds
        left, bottom = min(left, point_left), min(bottom, point_bottom)
        right, top = max(right, point_right), max(top, point_top)
    span = max(right-left, top-bottom)
    ax.set_xlim(left-span*.045, right+span*.045)
    ax.set_ylim(bottom-span*.045, top+span*.045)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if city_id == "new_york_city":
        transformer = Transformer.from_crs(4326, projection, always_xy=True)
        for name, coords in NYC_LABELS.items():
            x, y = transformer.transform(*coords)
            text = ax.text(x, y, name.upper(), ha="center", va="center", fontsize=7 if small else 8,
                color=INK, weight="bold", zorder=12)
            text.set_path_effects([patheffects.withStroke(linewidth=2.8, foreground="white", alpha=.9)])
        leader_start = transformer.transform(-74.017, 40.799)
        leader_end = transformer.transform(-73.963, 40.799)
        ax.plot([leader_start[0], leader_end[0]], [leader_start[1], leader_end[1]], color=INK, linewidth=.6, alpha=.65)
    ax.annotate("N", xy=(.956, .96), xytext=(.956, .88), xycoords="axes fraction",
        textcoords="axes fraction", ha="center", color=INK, fontsize=9,
        arrowprops={"arrowstyle": "-|>", "color": INK, "lw": 1}, zorder=15)
    scales = np.array([.25, .5, 1, 2, 5, 10, 20, 50, 100])*1000
    scale = scales[np.argmin(np.abs(scales - span*.16))]
    x0, y0 = right-scale-span*.04, bottom+span*.025
    ax.plot([x0, x0+scale], [y0, y0], color="white", linewidth=5, zorder=13)
    ax.plot([x0, x0+scale], [y0, y0], color=INK, linewidth=2, zorder=14)
    ax.text(x0+scale/2, y0+span*.014, f"{scale/1000:g} km", ha="center", color=INK,
        fontsize=7 if small else 8, bbox={"facecolor": "white", "edgecolor": "none", "alpha": .8, "pad": 1}, zorder=15)


def add_colorbar(fig, rectangle):
    cmap = matplotlib.colormaps["RdBu_r"].copy()
    cmap.set_under("#082e50"); cmap.set_over("#67001f")
    scalar = matplotlib.cm.ScalarMappable(norm=shared_norm(), cmap=cmap)
    bar = fig.colorbar(scalar, cax=fig.add_axes(rectangle), orientation="horizontal", extend="both", ticks=[.25, .5, 1, 2, 4])
    bar.ax.set_xticklabels(["0.25", "0.5", "1 · section mean", "2", "4"])
    bar.ax.tick_params(labelsize=10, length=3, colors=INK)
    bar.outline.set_visible(False)
    bar.ax.axvline(1, color=INK, linewidth=.8)
    bar.set_label("Relative PSSI eigenvector n × vᵢ  ·  shared logarithmic color scale", fontsize=11, labelpad=8, color=INK)


def provenance_caption(city_id, n, absent):
    scope = "70 NYC sections" if city_id == "new_york_city" else f"{n:,} populated city-limit tract portions"
    transit = "subway/SIR" if city_id == "new_york_city" else "bus/rail"
    text = (f"2020 Census · {scope} · approximate walking + local {transit} GTFS transit\n"
        "Uncapped, barrier-checked access/egress; automatic transfer links ≤400 m; explicit links/chained walks retained.\n"
        "Walking paths remain approximations, without verified street or bridge routing.")
    if absent.any():
        text += f"\n{absent.sum()} section(s) with no drawable land after water removal shown as diamonds; all scores retained."
    return text


def render_city(city_id):
    nodes, arrays, residuals, manifests, sources = load_city(city_id)
    frame, points, outline, absent, projection, geometry_sources, geometry_method = display_geometry(city_id, nodes)
    sources += geometry_sources
    out = OUTPUT / city_id
    out.mkdir(parents=True, exist_ok=True)
    n = len(nodes)
    values = n*arrays["right_eigenvectors"]
    caption = provenance_caption(city_id, n, absent)
    scope = "70 CITY SECTIONS" if city_id == "new_york_city" else f"{n:,} CITY-LIMIT TRACT PORTIONS"
    for k, (category, label) in enumerate(CATEGORIES.items()):
        fig = plt.figure(figsize=(10.5, 11.8), facecolor="white")
        fig.text(.065, .963, f"{CITIES[city_id].upper()} · {scope}", fontsize=11, weight="bold", color="#50677a")
        fig.text(.065, .922, label, fontsize=27, weight="bold", color=INK)
        fig.text(.065, .89, "PSSI right eigenvector · section mean = 1 · same colors in every city", fontsize=11, color="#50677a")
        ax = fig.add_axes([.07, .25, .86, .62])
        draw_map(ax, city_id, frame, points, outline, absent, projection, values[:, k])
        fig.text(.5, .225, range_text(values[:, k]), ha="center", fontsize=10, color=INK)
        add_colorbar(fig, [.22, .17, .56, .020])
        fig.text(.5, .102, "Arrow ends include values below 0.25 / above 4; exact scores are retained in the CSV.", ha="center", fontsize=9.2, color=INK)
        fig.text(.5, .079, "These are relative section eigenvector scores, not citywide λ or normalized PSSI.", ha="center", fontsize=9.4, color=INK)
        fig.text(.065, .025, caption, fontsize=8.6, color="#50677a", linespacing=1.6)
        fig.savefig(out/f"{category}_pssi_eigenvector.png", dpi=210, facecolor="white")
        plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(15, 16.3), facecolor="white")
    fig.subplots_adjust(left=.045, right=.955, top=.891, bottom=.218, hspace=.22, wspace=.05)
    fig.text(.055, .965, f"{CITIES[city_id]} · four PSSI eigenvectors", fontsize=25, weight="bold", color=INK)
    fig.text(.055, .932, f"{scope.lower()} · section mean = 1 · one shared color scale across all cities", fontsize=12, color="#50677a")
    for k, ((category, label), ax) in enumerate(zip(CATEGORIES.items(), axes.flat)):
        draw_map(ax, city_id, frame, points, outline, absent, projection, values[:, k], small=True)
        ax.set_title(label, loc="left", pad=11, fontsize=16, weight="bold", color=INK)
        ax.text(.5, -.042, range_text(values[:, k]), transform=ax.transAxes, ha="center", va="top", fontsize=8.1, color=INK)
    add_colorbar(fig, [.24, .158, .52, .018])
    fig.text(.5, .098, "Arrow ends include values below 0.25 / above 4; exact scores are retained in the CSV.", ha="center", fontsize=10, color=INK)
    fig.text(.5, .077, "Colors show relative eigenvector components, not citywide λ or normalized PSSI.", ha="center", fontsize=10, color=INK)
    fig.text(.055, .024, caption + "\nHispanic is any race and overlaps with the race-alone categories.", fontsize=9, color="#50677a", linespacing=1.6)
    fig.savefig(out/"four_category_pssi_eigenvectors.png", dpi=200, facecolor="white")
    plt.close(fig)
    table = nodes[["matrix_index", "tract_geoid", "latitude", "longitude", "population"]].copy().rename(columns={"tract_geoid": "section_id"})
    table["no_land_display_geometry"] = absent
    table.insert(0, "city", CITIES[city_id]); table.insert(0, "city_id", city_id)
    long = []
    diagnostics = {}
    for k, (category, label) in enumerate(CATEGORIES.items()):
        row = table.copy()
        row["category"] = category
        row["category_population"] = arrays["counts"][:, k]
        row["category_proportion"] = arrays["proportions"][:, k]
        row["raw_eigenvector_v"] = arrays["right_eigenvectors"][:, k]
        row["relative_eigenvector_n_v"] = values[:, k]
        row["below_color_scale"] = values[:, k] < VMIN
        row["above_color_scale"] = values[:, k] > VMAX
        row["citywide_eigenvalue_lambda"] = arrays["eigenvalues"][k]
        long.append(row)
        diagnostics[category] = {"eigenvalue": float(arrays["eigenvalues"][k]), "vector_sum": float(arrays["right_eigenvectors"][:, k].sum()),
            "relative_score_mean": float(values[:, k].mean()), "minimum": float(values[:, k].min()), "maximum": float(values[:, k].max()),
            "below_color_scale": int((values[:, k] < VMIN).sum()), "above_color_scale": int((values[:, k] > VMAX).sum()),
            "zero_components": int((values[:, k] == 0).sum()), "max_abs_eigenpair_residual": residuals[k]}
    table = pd.concat(long, ignore_index=True)
    table.to_csv(out/"section_eigenvectors.csv", index=False)
    metadata = {"created_utc": datetime.now(timezone.utc).isoformat(), "city_id": city_id, "city": CITIES[city_id],
        "section_count": n, "population": int(nodes.population.sum()), "geography": scope.lower(),
        "map_value": "n times the saved right Perron eigenvector component; vector sums to one, arithmetic mean of mapped values is one",
        "color_scale": {"shared_across_all_cities_and_categories": True, "transform": "log", "minimum": VMIN, "maximum": VMAX,
            "neutral_value": 1, "out_of_range": "Explicitly saturated with colorbar extension triangles and counts/ranges on every panel; exact values remain in CSV", "zeros": "Below-scale color; counted separately, not dropped"},
        "category_diagnostics": diagnostics, "display_projection": projection.to_string(), "display_geometry": geometry_method,
        "sections_without_land_display_geometry": nodes.loc[absent, "tract_geoid"].tolist(),
        "checks": ["Exact node/geometry/eigenvector ID ordering", "Counts and population match saved nodes", "P rows and eigenvectors normalized", "A=P diag(p)", "A v=lambda v within 1e-9"],
        "sources": [{"path": str(path.relative_to(ROOT)), "sha256": sha256(path)} for path in sources],
        "model_manifests": manifests, "network_requests": 0, "routes_api_calls": 0}
    (out/"metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    write_city_index(city_id, table, metadata)
    print(json.dumps({"city": city_id, "maps": 5, "nodes": n, "max_residual": max(residuals), "no_land_shapes": int(absent.sum())}), flush=True)
    return metadata


STYLE = "body{margin:0;background:#f2f6f8;color:#172b3a;font:16px/1.55 system-ui,sans-serif}main{max-width:1400px;margin:auto;padding:30px}h1{font-size:32px;line-height:1.2}h2{font-size:21px}p{max-width:1150px}.note{background:white;padding:20px;border-left:4px solid #385e77}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:22px}article{background:white;padding:18px;border-radius:8px}img{display:block;width:100%;height:auto}.scroll{max-height:500px;overflow:auto;background:white}table{border-collapse:collapse;width:100%}td,th{text-align:left;padding:10px;border-bottom:1px solid #dbe3e8}th{position:sticky;top:0;background:#e9f0f5}td{font-variant-numeric:tabular-nums}a{color:#155990}.downloads{display:flex;gap:20px;flex-wrap:wrap}@media(max-width:750px){.grid{grid-template-columns:1fr}main{padding:20px}}"
EXPLANATION = "<div class=\"note\"><strong>Reading the colors:</strong> each map shows n × vᵢ, where n is the city’s section count and the right eigenvector sums to one. The arithmetic section mean is 1. The common logarithmic color scale spans 0.25–4 in every panel and city. Values outside that range use the arrow-end colors; observed minima and maxima (rounded for display) and saturation counts are printed on each panel. Zero numerical components use the below-range color and are flagged. Full-precision values remain in the CSVs. These are relative section scores, not citywide eigenvalues or normalized PSSI.</div>"


def page_start(title):
    return f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{escape(title)}</title><style>{STYLE}</style></head><body><main><h1>{escape(title)}</h1>'


def write_city_index(city_id, table, metadata):
    html = page_start(f"{CITIES[city_id]} · four PSSI eigenvector maps")
    html += '<p><a href="../index.html">All city maps</a></p>' + EXPLANATION
    html += '<p class="downloads"><a href="four_category_pssi_eigenvectors.png">Combined PNG</a><a href="section_eigenvectors.csv">Section values CSV</a><a href="metadata.json">Validation and provenance</a></p>'
    html += '<img src="four_category_pssi_eigenvectors.png" alt="Four revised PSSI eigenvector maps">'
    html += '<h2>Individual maps</h2><p class="downloads">' + ''.join(f'<a href="{category}_pssi_eigenvector.png">{escape(label)}</a>' for category, label in CATEGORIES.items()) + '</p>'
    html += '<p>2020 Census populations; locally computed walking and GTFS transit. Walking access and egress have no distance cutoff, while water-chord checks remain. Automatic transfer links connect stops within 400 meters; explicit GTFS links and chained walking transfers are retained, so 400 meters is not a total transfer-walking limit. These approximate paths are not verified street or bridge routes. No Routes API calls were made. Removing water from the map changes only display geometry, not counts, assignments or scores. Hispanic overlaps with the race-alone categories.</p>'
    html += '<h2>Section values</h2><div class="scroll">' + table[["section_id", "category", "population", "raw_eigenvector_v", "relative_eigenvector_n_v", "below_color_scale", "above_color_scale"]].to_html(index=False, border=0, float_format=lambda x: f"{x:.6g}") + '</div></main></body></html>'
    (OUTPUT/city_id/"index.html").write_text(html)


def write_gallery():
    completed = []
    for city in CITIES:
        path = OUTPUT/city/"metadata.json"
        if path.exists() and (OUTPUT/city/"four_category_pssi_eigenvectors.png").exists():
            completed.append(json.loads(path.read_text()))
    archive_ready = len(completed) == len(CITIES)
    if archive_ready:
        temporary = OUTPUT/"combined_map_overviews.zip.tmp"
        with ZipFile(temporary, "w", compression=ZIP_STORED) as archive:
            for item in completed:
                city = item["city_id"]
                archive.write(OUTPUT/city/"four_category_pssi_eigenvectors.png", arcname=f"{city}_pssi_eigenvectors.png")
        temporary.replace(OUTPUT/"combined_map_overviews.zip")
    html = page_start("PSSI eigenvector maps · walking access to transit") + EXPLANATION
    html += '<p>Four categories per city: White alone, Black alone, Hispanic any race and Asian alone. New York City retains its 70 sections; other cities use individual populated 2020 city-limit tract portions. Section size and geometry differ between cities. Higher colors indicate higher relative eigenvector components within that city/category, not a ranking of citywide segregation.</p>'
    html += '<p class="downloads"><a href="all_city_section_eigenvectors.csv">All section values CSV</a>'
    if archive_ready:
        html += '<a href="combined_map_overviews.zip">Download all 12 combined maps (ZIP)</a>'
    html += '</p>'
    html += '<div class="grid">'
    all_tables = []
    for item in completed:
        city, title = item["city_id"], item["city"]
        html += f'<article><h2><a href="{city}/index.html">{escape(title)}</a></h2><p>{item["section_count"]:,} sections · {item["population"]:,} residents</p><a href="{city}/index.html"><img loading="lazy" src="{city}/four_category_pssi_eigenvectors.png" alt="Four PSSI eigenvector maps for {escape(title)}"></a><p><a href="{city}/section_eigenvectors.csv">Exact section values</a> · <a href="{city}/four_category_pssi_eigenvectors.png">Combined PNG</a></p></article>'
        all_tables.append(pd.read_csv(OUTPUT/city/"section_eigenvectors.csv", dtype={"section_id": str}))
    html += '</div><p>Maps use saved, recomputed model arrays. No API calls or downloads occur during map creation. Details, saturation counts, geometry exceptions and source hashes are recorded in each city’s metadata.</p></main></body></html>'
    (OUTPUT/"index.html").write_text(html)
    if all_tables:
        pd.concat(all_tables, ignore_index=True).to_csv(OUTPUT/"all_city_section_eigenvectors.csv", index=False)
    (OUTPUT/"manifest.json").write_text(json.dumps({"cities_completed": [x["city_id"] for x in completed], "city_count": len(completed), "png_count": len(completed)*5,
        "color_scale": {"min": VMIN, "max": VMAX, "transform": "log", "common_across_cities": True, "explicit_saturation": True}, "routes_api_calls": 0}, indent=2) + "\n")


def build(cities=None):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    for city in cities or CITIES:
        render_city(city)
        write_gallery()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cities", nargs="+", choices=list(CITIES))
    args = parser.parse_args()
    build(args.cities)
