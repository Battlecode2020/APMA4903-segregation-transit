"""Map the four verified NYC PSSI right eigenvectors using saved local data.

No networking, downloads, Routes API calls, or model recomputation. Colors show
n*v_i, with arithmetic mean one across the 70 sections for each category.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from html import escape
import json
import os
from pathlib import Path

import geopandas as gpd
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "outputs/nyc_routes/eigenvector_maps/.matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.colors import FuncNorm
import numpy as np
import pandas as pd
from pyproj import Transformer
import shapely

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/nyc_routes"
MODEL = BASE / "cost_model/estimated"
OUTPUT = BASE / "eigenvector_maps"
CATEGORIES = {
    "white_alone": "White alone",
    "black_alone": "Black alone",
    "hispanic_latino_any_race": "Hispanic · any race",
    "asian_alone": "Asian alone",
}
PROJECTION = "EPSG:32118"  # NAD83 / New York Long Island, meters.
BOROUGH_LABELS = {
    "Bronx": (-73.862, 40.866),
    "Manhattan": (-74.033, 40.799),
    "Queens": (-73.802, 40.734),
    "Brooklyn": (-73.950, 40.631),
    "Staten Island": (-74.164, 40.574),
}
INK = "#172b3a"
PAGE = "#ffffff"
WATER = "#edf3f7"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def polygonal_only(shape):
    """Remove zero-area intersection artifacts before drawing borough outlines."""
    if shape.geom_type == "Polygon":
        return shape
    polygons = []
    for part in getattr(shape, "geoms", []):
        if part.geom_type == "Polygon":
            polygons.append(part)
        elif hasattr(part, "geoms"):
            polygons.append(polygonal_only(part))
    return shapely.union_all(polygons)


def load_verified():
    """Check ID ordering, population alignment, defining A formula and eigenpairs."""
    with np.load(MODEL / "four_category_analysis.npz", allow_pickle=False) as saved:
        arrays = {name: saved[name] for name in saved.files}
    zones = pd.read_csv(BASE / "zones.csv", dtype={"zone_id": str})
    geometry = gpd.read_file(BASE / "zones.geojson")
    model_manifest = json.loads((BASE / "cost_model/manifest.json").read_text())
    n = len(zones)
    require(n == 70 and geometry.zone_id.is_unique and zones.zone_id.is_unique, "Expected 70 unique NYC zones")
    require(set(geometry.zone_id) == set(zones.zone_id), "Geometry and saved zones have different section IDs")
    require(zones.matrix_index.tolist() == list(range(n)), "zones.csv order differs from matrix_index")
    require(np.array_equal(arrays["zone_ids"], zones.zone_id), "Saved eigenvectors and zone rows are misaligned")
    require(arrays["group_names"].tolist() == list(CATEGORIES), "Unexpected demographic category order")
    require(np.array_equal(arrays["population"], zones.TOTPOP), "Population alignment differs")
    geometry = geometry.set_index("zone_id").loc[zones.zone_id].reset_index()
    require(np.array_equal(geometry.matrix_index, zones.matrix_index), "GeoJSON matrix_index differs from eigenvectors")
    require(np.array_equal(geometry.TOTPOP, arrays["population"]), "Geometry population differs from model")
    require(geometry.geometry.notna().all() and (~geometry.geometry.is_empty).all(), "Empty section geometry")
    require(geometry.geometry.is_valid.all(), "Invalid section geometry")
    P, A, p, v, eigenvalues = (arrays[key] for key in ("P", "A", "proportions", "right_eigenvectors", "eigenvalues"))
    require(P.shape == (n, n) and A.shape == (4, n, n) and v.shape == (n, 4), "Unexpected matrix dimensions")
    require(np.isfinite(v).all() and (v >= 0).all(), "Eigenvector entries must be finite and nonnegative")
    require(np.allclose(v.sum(axis=0), 1, atol=1e-12, rtol=0), "Right eigenvector columns must sum to one")
    require(np.allclose(P.sum(axis=1), 1, atol=1e-12, rtol=0), "P rows must sum to one")
    require(np.allclose(arrays["counts"] / arrays["population"][:, None], p, atol=1e-14, rtol=0), "Demographic proportions differ from counts")
    residuals = []
    for k, name in enumerate(CATEGORIES):
        require(np.allclose(A[k], P * p[:, k][None, :], atol=1e-14, rtol=0), f"A definition failed for {name}")
        residual = float(np.max(np.abs(A[k] @ v[:, k] - eigenvalues[k] * v[:, k])))
        require(residual < 1e-10, f"Perron eigenpair failed for {name}: {residual}")
        radius = float(np.max(np.abs(np.linalg.eigvals(A[k]))))
        require(np.isclose(radius, eigenvalues[k], atol=1e-10, rtol=0), f"Saved eigenvalue is not the spectral radius for {name}")
        residuals.append(residual)
    geometry = geometry.to_crs(PROJECTION)
    # Census zones include municipal water areas. Clip only their cartographic
    # display using the saved land-block source, retaining every ID and score.
    blocks = gpd.read_file(ROOT / "data/raw/tl_2020_36_tabblock20.shp",
        where="COUNTYFP20 IN ('005','047','061','081','085')",
        columns=["COUNTYFP20", "ALAND20", "geometry"]).to_crs(PROJECTION)
    require(len(blocks) > 0, "Saved NYC land-block selection is empty")
    land = blocks.loc[pd.to_numeric(blocks.ALAND20) > 0].geometry.union_all()
    geometry.geometry = geometry.geometry.intersection(land).map(polygonal_only)
    require((~geometry.geometry.is_empty).all(), "Land display clipping removed a section")
    require(geometry.geometry.is_valid.all(), "Land-clipped section geometry is invalid")
    return geometry, arrays, model_manifest, residuals


def shared_norm(values):
    """Symmetric log-ratio colors around mean one; no clipped observed values."""
    require((values > 0).all(), "Logarithmic relative-score map requires positive values")
    bound = max(1.0, float(values.max()), float(1 / values.min()))
    bound = np.ceil(bound * 2) / 2  # shared rounded bound, currently 4.5.
    def safe_log(values):
        return np.log(np.maximum(values, np.finfo(float).tiny))
    return FuncNorm((safe_log, np.exp), vmin=1 / bound, vmax=bound), float(bound)


def map_panel(ax, geometry, relative_scores, norm, *, title=None, small=False):
    ax.set_facecolor(WATER)
    geometry.plot(ax=ax, column=relative_scores, cmap="RdBu_r", norm=norm,
                  edgecolor="#ffffff", linewidth=.5 if not small else .35, antialiased=True)
    boroughs = geometry.dissolve(by="borough")
    boroughs.boundary.plot(ax=ax, color="#344958", linewidth=.85 if not small else .65, alpha=.8)
    transformer = Transformer.from_crs(4326, PROJECTION, always_xy=True)
    for name, location in BOROUGH_LABELS.items():
        x, y = transformer.transform(*location)
        text = ax.text(x, y, name.upper(), color=INK, fontsize=8 if small else 9,
                       weight="bold", ha="center", va="center", zorder=10)
        text.set_path_effects([patheffects.withStroke(linewidth=3, foreground="white", alpha=.9)])
    # A leader places Manhattan's label outside its narrow shape.
    start = transformer.transform(-74.017, 40.799)
    end = transformer.transform(-73.963, 40.799)
    ax.plot([start[0], end[0]], [start[1], end[1]], color=INK, linewidth=.65, alpha=.65)
    left, bottom, right, top = geometry.total_bounds
    pad = 1900
    ax.set_xlim(left-pad, right+pad)
    ax.set_ylim(bottom-pad, top+pad)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    # Projection uses meters; the five-kilometer scale is in projected units.
    x0, y0 = right-9000, bottom+1700
    ax.plot([x0, x0+5000], [y0, y0], color=INK, linewidth=2)
    ax.text(x0+2500, y0+500, "5 km", color=INK, fontsize=7 if small else 8, ha="center")
    ax.annotate("N", xy=(.955, .96), xytext=(.955, .88), xycoords="axes fraction",
                textcoords="axes fraction", ha="center", color=INK, fontsize=9,
                arrowprops={"arrowstyle": "-|>", "color": INK, "lw": 1})
    if title:
        ax.set_title(title, loc="left", pad=10, fontsize=16 if small else 19, color=INK, weight="bold")


def colorbar(fig, cax, norm, *, compact=False):
    scalar = matplotlib.cm.ScalarMappable(norm=norm, cmap="RdBu_r")
    bar = fig.colorbar(scalar, cax=cax, orientation="horizontal", ticks=[.25, .5, 1, 2, 4])
    bar.ax.set_xticklabels(["0.25", "0.5", "1 · section mean", "2", "4"])
    bar.ax.tick_params(labelsize=10 if compact else 11, length=3, colors=INK)
    bar.outline.set_visible(False)
    bar.ax.axvline(1, color=INK, linewidth=.8)
    bar.set_label("Relative PSSI eigenvector: 70 × vᵢ  ·  logarithmic color scale",
                  fontsize=11 if compact else 12, color=INK, labelpad=8)


def render_maps(geometry, arrays, model_manifest, norm):
    n = len(geometry)
    values = n * arrays["right_eigenvectors"]
    proxy_count = model_manifest["variants"]["estimated"]["selected_transit_proxy"]
    service_date = model_manifest["source"]["service_date"]
    provenance = (f"2020 Census populations · subway/SIR schedule {service_date} · approximate walking\n"
                  f"Estimated transit includes {proxy_count:,} proxy pairs of {n*(n-1):,} directed nonlocal pairs; no bus/ferry network.")
    for k, (name, label) in enumerate(CATEGORIES.items()):
        fig = plt.figure(figsize=(10.5, 11.5), facecolor=PAGE)
        fig.text(.07, .96, "NYC · 70 CITY SECTIONS", fontsize=11, color="#50677a", weight="bold")
        fig.text(.07, .918, label, fontsize=27, color=INK, weight="bold")
        fig.text(.07, .884, "PSSI right eigenvector · same color scale across all four categories", fontsize=11, color="#50677a")
        ax = fig.add_axes([.07, .22, .86, .64])
        map_panel(ax, geometry, values[:, k], norm)
        colorbar(fig, fig.add_axes([.22, .161, .56, .020]), norm)
        fig.text(.5, .089, "1 = section average   ·   2 = twice the average   ·   0.5 = half the average",
                 fontsize=10, ha="center", color=INK)
        fig.text(.5, .065, "Colors show relative eigenvector components, not citywide λ or normalized PSSI.",
                 fontsize=9.5, ha="center", color=INK)
        fig.text(.07, .022, provenance, fontsize=8.6, color="#50677a", linespacing=1.6)
        fig.savefig(OUTPUT / f"{name}_pssi_eigenvector.png", dpi=230, facecolor=PAGE)
        plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(15, 16), facecolor=PAGE)
    fig.subplots_adjust(left=.045, right=.955, top=.897, bottom=.197, hspace=.12, wspace=.03)
    fig.text(.055, .966, "NYC segregation · the four PSSI eigenvectors", fontsize=26, color=INK, weight="bold")
    fig.text(.055, .934, "70 sections · each category has section mean = 1 · one shared color scale", fontsize=13, color="#50677a")
    for k, ((name, label), ax) in enumerate(zip(CATEGORIES.items(), axes.flat)):
        map_panel(ax, geometry, values[:, k], norm, title=label, small=True)
    colorbar(fig, fig.add_axes([.24, .145, .52, .018]), norm, compact=True)
    fig.text(.5, .09, "1 = section average   ·   2 = twice the average   ·   0.5 = half the average", fontsize=11, ha="center", color=INK)
    fig.text(.5, .067, "Colors show relative eigenvector components, not citywide λ or normalized PSSI.", fontsize=10, ha="center", color=INK)
    fig.text(.055, .021, provenance + "\nHispanic is any race and overlaps with the race-alone categories.", fontsize=9.5, color="#50677a", linespacing=1.65)
    fig.savefig(OUTPUT / "nyc_four_category_pssi_eigenvectors.png", dpi=210, facecolor=PAGE)
    plt.close(fig)


def export_tables(geometry, arrays):
    rows = []
    wide = geometry[["matrix_index", "zone_id", "borough", "TOTPOP"]].drop(columns="geometry", errors="ignore").rename(columns={"TOTPOP": "population"})
    for k, (category, label) in enumerate(CATEGORIES.items()):
        part = wide.copy()
        part["category"] = category
        part["category_label"] = label
        part["category_population"] = arrays["counts"][:, k]
        part["category_proportion"] = arrays["proportions"][:, k]
        part["pssi_right_eigenvector_v"] = arrays["right_eigenvectors"][:, k]
        part["relative_pssi_eigenvector_n_times_v"] = len(geometry) * arrays["right_eigenvectors"][:, k]
        part["citywide_pssi_eigenvalue_lambda"] = arrays["eigenvalues"][k]
        rows.append(part)
        wide[f"{category}_v"] = arrays["right_eigenvectors"][:, k]
        wide[f"{category}_relative_n_v"] = len(geometry) * arrays["right_eigenvectors"][:, k]
    long = pd.concat(rows, ignore_index=True)
    # Every long-table row has one category; keep the parallel wide columns only
    # in the separate section-by-section file.
    long = long[["matrix_index", "zone_id", "borough", "population", "category", "category_label",
                 "category_population", "category_proportion", "pssi_right_eigenvector_v",
                 "relative_pssi_eigenvector_n_times_v", "citywide_pssi_eigenvalue_lambda"]]
    long.to_csv(OUTPUT / "nyc_section_eigenvectors.csv", index=False)
    wide.to_csv(OUTPUT / "nyc_section_eigenvectors_wide.csv", index=False)
    return long


def write_html(table, metadata):
    cards = "".join(f'<article><h2>{escape(label)}</h2><a href="{name}_pssi_eigenvector.png"><img src="{name}_pssi_eigenvector.png" alt="NYC relative PSSI eigenvector for {escape(label)}"></a></article>'
                    for name, label in CATEGORIES.items())
    table_html = table.rename(columns={"zone_id": "Section", "borough": "Borough", "category_label": "Category",
        "pssi_right_eigenvector_v": "Raw eigenvector vᵢ", "relative_pssi_eigenvector_n_times_v": "Relative score 70 × vᵢ"})[
        ["Section", "Borough", "Category", "Raw eigenvector vᵢ", "Relative score 70 × vᵢ"]].to_html(index=False, border=0, float_format=lambda x: f"{x:.6f}")
    html = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>NYC PSSI eigenvector maps</title>
<style>body{margin:0;background:#f2f6f8;color:#172b3a;font:16px/1.55 system-ui,sans-serif}main{max-width:1400px;margin:auto;padding:32px}h1{font-size:34px;line-height:1.2}h2{font-size:21px}p{max-width:1100px}.note{background:white;padding:20px;border-left:4px solid #385e77}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:22px}article{background:white;padding:18px;border-radius:8px}img{display:block;width:100%;height:auto}.scroll{max-height:500px;overflow:auto;background:white}table{border-collapse:collapse;width:100%}td,th{text-align:left;padding:10px;border-bottom:1px solid #dbe3e8}th{position:sticky;top:0;background:#e9f0f5}td{font-variant-numeric:tabular-nums}a{color:#155990}.downloads{display:flex;gap:24px;flex-wrap:wrap}@media(max-width:750px){.grid{grid-template-columns:1fr}main{padding:20px}}</style></head><body><main>
<h1>NYC · four PSSI eigenvector maps</h1>
<p>White alone, Black alone, Hispanic any race, and Asian alone across the existing 70 city sections.</p>
<div class="note"><strong>What the colors mean:</strong> each map shows 70 × vᵢ, where v is the saved right Perron eigenvector with sum vᵢ = 1. The mean section score is therefore 1. A score of 2 is twice this average, and 0.5 is half. All maps share the same logarithmic color scale centered on 1. These are relative section scores; they are not the citywide eigenvalue λ, probabilities, or (λ−K)/(1−K).</div>
<p class="downloads"><a href="nyc_four_category_pssi_eigenvectors.png">Combined four-map PNG</a><a href="nyc_section_eigenvectors.csv">Section data CSV</a><a href="nyc_section_eigenvectors_wide.csv">Wide section data CSV</a><a href="metadata.json">Validation and provenance</a></p>'''
    html += f'<div class="grid">{cards}</div>'
    html += '''<h2>Model provenance</h2><p>These maps use the existing saved NYC model: 2020 Census counts, approximate walking, subway/SIR GTFS service on October 20, 2026 at 08:00, and the existing estimated-transit completion. There are 1,284 selected hypothetical transit-proxy pairs among 4,830 directed nonlocal pairs. Bus and ferry networks are not implemented in this NYC model. No travel data or eigenvectors were recomputed to create the maps, and no APIs were called.</p>
<p>Model parameters: β = 0.1 per dollar; value of time = $15/hour; local encounter floor c₀ = $1.25; standard paid transit fare = $3. White, Black and Asian alone include Hispanic residents, so the four categories overlap. Borough outlines and labels provide geographic context; the CSV identifies every section.</p>
<p>Validation checked section IDs and ordering, population alignment, A = P diag(p), sum v = 1, A v = λ v, and λ equal to the matrix spectral radius. Raw eigenvector components are preserved in the CSV; the factor of 70 changes only the display scale. Coastlines are drawn by clipping the saved section polygons to the union of saved NYC Census blocks with positive land area, matching the land source used in the earlier subway map. This display operation changes no section assignments or scores.</p>
<h2>Section values</h2>'''
    html += f'<div class="scroll">{table_html}</div></main></body></html>'
    (OUTPUT / "index.html").write_text(html)


def build():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})
    geometry, arrays, model_manifest, residuals = load_verified()
    values = len(geometry) * arrays["right_eigenvectors"]
    norm, bound = shared_norm(values)
    render_maps(geometry, arrays, model_manifest, norm)
    table = export_tables(geometry, arrays)
    sources = [MODEL / "four_category_analysis.npz", BASE / "zones.csv", BASE / "zones.geojson", BASE / "cost_model/manifest.json",
               ROOT / "data/raw/tl_2020_36_tabblock20.shp", ROOT / "data/raw/tl_2020_36_tabblock20.dbf"]
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "section_count": len(geometry),
        "total_population": int(arrays["population"].sum()), "categories": CATEGORIES,
        "map_value": "70 * right Perron eigenvector component v_i; sum(v)=1, mean(70*v)=1 for each category",
        "not_mapped": ["Citywide PSSI eigenvalue", "Population-share normalized scalar PSSI", "Race proportion", "Fryer SSI"],
        "projection": PROJECTION,
        "display_geometry": "Saved sections intersected with union of saved NYC Census 2020 blocks having ALAND20 > 0. Cartographic water removal only; no changes to assignments, counts, coordinates or scores. Land-containing blocks may include some shoreline water.",
        "common_color_scale": {"transform": "natural logarithm", "minimum": 1/bound, "maximum": bound, "neutral_value": 1, "cmap": "RdBu_r", "observed_values_clipped": False},
        "category_validation": {name: {"eigenvalue": float(arrays["eigenvalues"][k]), "eigenvector_sum": float(arrays["right_eigenvectors"][:,k].sum()),
            "relative_score_mean": float(values[:,k].mean()), "relative_score_min": float(values[:,k].min()), "relative_score_max": float(values[:,k].max()),
            "max_abs_eigenpair_residual": residuals[k]} for k, name in enumerate(CATEGORIES)},
        "checks": ["70 unique geometry and model IDs", "Geometry reordered by exact saved zone_id; matrix_index and populations match", "A equals P diag(p)", "Right vectors nonnegative and sum to one", "A v equals lambda v within 1e-10", "Eigenvalue matches spectral radius within 1e-10"],
        "sources": [{"path": str(path.relative_to(ROOT)), "sha256": sha256(path)} for path in sources],
        "model_parameters": model_manifest["parameters"], "travel_provenance": model_manifest["source"],
        "estimated_travel_counts": model_manifest["variants"]["estimated"], "limitations": model_manifest["limitations"],
        "network_requests": 0, "routes_api_calls": 0,
    }
    (OUTPUT / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    write_html(table, metadata)
    print(json.dumps({"output": str(OUTPUT), "png_maps": 5, "csv_rows": len(table), "maximum_eigenpair_residual": max(residuals)}, indent=2))


if __name__ == "__main__":
    build()
