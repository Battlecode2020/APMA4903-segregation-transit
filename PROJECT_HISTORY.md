# New York 2020 Census dual graphs

A 2020 counterpart to Daryl DeFord’s Census dual-graph collection, built with **GerryChain** and **maup** from the supplied `tl_2020_36_all.zip`. Open `outputs/ny_2020/index.html` for the formatted catalog, maps, data dictionary, validation results, and file links.

The archive contains both 2010 and 2020 layers. This project uses the **2020 geography**, preserving its `GEOID20` identifiers and adding `BLOCKID20` for blocks. It does not relabel 2020 data with 2010 field names. Statewide layers are used once; the duplicate county-specific copies are not combined with them.

## Files

Each of `blocks`, `block_groups`, `tracts`, `county_subdivisions`, and `counties` contains:

- `graph.json`: undirected NetworkX adjacency JSON, loadable with GerryChain.
- `nodes.csv`: one row per unit, with geographic and demographic attributes. `id` is the zero-based row index in the original statewide shapefile. `component` and `degree` are additional table diagnostics.
- `edges.csv`: one row per undirected edge, with source/target IDs, GEOIDs, and shared perimeter.
- `validation.json`: counts, connectivity diagnostics, and validation outcomes.
- `warnings.json`: captured library warnings; islands are preserved.
- `block_assignment.csv.gz` for larger units: every block’s maup assignment, checked against Census identifiers.

`manifest.json` records input checksums and package versions. `data_dictionary.csv` defines the common graph attributes and all 75 demographic columns. The original Census fields are retained; their detailed definitions are in the source `.shp.ea.iso.xml` files under `data/raw`.

## Load a graph

```python
from gerrychain import Graph

graph = Graph.from_json("outputs/ny_2020/tracts/graph.json")
```

Or use NetworkX directly:

```python
import json
import networkx as nx

with open("outputs/ny_2020/tracts/graph.json") as f:
    graph = nx.adjacency_graph(json.load(f))
```

When reading CSV files, keep GEOIDs as strings. For example:

```python
import pandas as pd
nodes = pd.read_csv("outputs/ny_2020/blocks/nodes.csv",
                    dtype={"GEOID20": str, "BLOCKID20": str})
```

## Methods and units

1. Read the five statewide 2020 TIGER layers from the supplied archive. Preserve source row order and reject invalid or empty geometries.
2. Join 2020 PL 94-171 segments to the geographic header by `LOGRECNO`. Match summary-level 750 records to blocks by 15-digit GEOID. Race and Hispanic-origin counts come from P1/P2; voting-age counts from P3/P4. Hispanic race counts are derived as all-race counts minus the corresponding non-Hispanic counts.
3. Replace the original archive’s blank `UR20`, `UACE20`, and `UATYPE20` block attributes with the official updated TIGER2020 block attributes, joined by GEOID. Retain the supplied archive’s original geometry. The updated file’s `POP20` provides an independent join check.
4. Use `maup.assign` to assign every block to each larger geography. Assignments use the common native NAD83 coordinate system to preserve the original nesting. Verify every result against Census hierarchy codes, including county subdivisions from the PL geographic header. Sum block counts, including separate urban and rural sums.
5. Project geometries to **EPSG:5070 (NAD83 / Conus Albers)** and build GerryChain rook graphs. Only positive-length shared boundaries produce edges. Do not add ferry, bridge, nearest-neighbor, or other artificial links. Polygon water areas are retained as supplied.
6. Store `area` as **Census ALAND20 in square metres**, matching DeFord’s land-area convention; store full projected polygon area separately as `geometry_area`. Store geometric centroids as `C_X` and `C_Y` in EPSG:5070 metres. These are distinct from the Census internal points `INTPTLAT20` / `INTPTLON20` in degrees. Perimeters use projected metres, so they are consistent but not geodesic lengths.
7. Use GerryChain boundary flags and exterior perimeters relative to the union of that layer’s polygons, including coastlines and any holes. A boundary flag can be true at a point touch even when exterior perimeter is zero. No boundary snapping, simplification, or repair is applied to analytical geometries. Display maps alone are simplified.

The unsuffixed demographic columns refer to the 2020 Census, not current population estimates. `U` and `R` prefixes mean sums over urban and rural **blocks**, respectively; mixed larger units have both. `VAP` means age 18+, not citizen voting-age population. The P.L. 94-171 counts are the Census release, without a separate New York prisoner-reallocation adjustment. These are a 2020 adaptation of DeFord’s schema, not a byte-for-byte reproduction of his 2010 files.

## Validation

The build stops on missing/duplicate GEOIDs, incomplete urban labels, invalid polygons, assignment discrepancies, negative demographic counts, failed race-sum identities, overlaps reported by GerryChain, nonpositive shared perimeters, or missing population records. For all five levels, all 25 base demographic totals are compared with the independently published Census records for each unit. All 75 statewide sums must be conserved, and every `U` + `R` decomposition must equal its base count. Components and isolates are reported rather than silently connected.

## Reproduce

Use Python 3.11. The environment is isolated in `.venv`.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/prepare_inputs.py /path/to/tl_2020_36_all.zip
.venv/bin/python scripts/build_graphs.py
.venv/bin/python scripts/make_catalog.py
.venv/bin/python scripts/verify_outputs.py
.venv/bin/python scripts/package_outputs.py
```

The preparation step downloads about 380 MB of Census supplements. The full block graph can take several minutes. To rebuild one level, use e.g. `scripts/build_graphs.py --levels tracts`. Raw data, generated outputs, and the virtual environment are ignored by Git; the ZIP deliverable includes the output data and reproducibility files but excludes the large raw inputs and environment.

GeoPandas is pinned to 1.1.1 because maup 2.1.0 references an option removed in GeoPandas 1.2.0.

## Sources

- [DeFord’s dual-graph collection](https://people.csail.mit.edu/ddeford/dual_graphs.html) and [column definitions](https://people.csail.mit.edu/ddeford/data_cols.html).
- [GerryChain](https://github.com/mggg/GerryChain) and [maup](https://github.com/mggg/maup).
- [2020 TIGER/Line technical documentation](https://www.census.gov/programs-surveys/geography/technical-documentation/complete-technical-documentation/tiger-geo-line.2020.html).
- [New York 2020 PL 94-171 download](https://www2.census.gov/programs-surveys/decennial/2020/data/01-Redistricting_File--PL_94-171/New_York/).
- [Official PL layout scripts](https://www2.census.gov/programs-surveys/decennial/rdo/about/2020-census-program/Phase3/SupportMaterials/2020PL_R_import_scripts.zip).
- [Updated 2020 block shapefiles](https://www2.census.gov/geo/tiger/TIGER2020/TABBLOCK20/) and [Census urban/rural guidance](https://www.census.gov/programs-surveys/geography/guidance/geo-areas/urban-rural.html).

## Coordinate-based social weights

`scripts/social_weight.py` implements
`f(i -> j) = exp(-beta * max(c0, min_m(p_m + omega * t_m)))`
using Google Routes API HTTP requests, with no additional Python dependencies.
Inputs are two `(latitude, longitude)` pairs in WGS84 degrees. **Do not pass
`C_X`/`C_Y`**, which are EPSG:5070 metres. Transform your chosen representative
points to WGS84 first; Census internal points are another explicit modeling choice,
not population-weighted centroids.

Enable Google Routes API and billing in your Google Cloud project, and configure
`GOOGLE_MAPS_API_KEY` locally. Do not commit the key. Each nonlocal pair makes one
billable request per requested mode; this is a pairwise evaluator, not a statewide
all-pairs job.

```python
from scripts.social_weight import social_weight

result = social_weight(
    (40.8075, -73.9626), (40.7506, -73.9935),
    beta=0.1,           # 1/USD; illustrative, not calibrated
    omega=15.0,         # USD/hour; illustrative
    c0=15.0 * 5 / 60,   # fixed local floor: five minutes valued at omega
    modes=("WALK", "TRANSIT"),
)
print(result.f, result.cost_usd, result.best_mode)
print(result.alternatives)
```

Equivalent command:

```bash
python3 scripts/social_weight.py 40.8075 -73.9626 40.7506 -73.9935 \
  --beta 0.1 --omega 15 --c0 1.25 --modes WALK TRANSIT
python3 -m unittest discover -s tests -v
```

Mode assumptions:

- `WALK`: zero out-of-pocket cost.
- `BICYCLE`: zero by default (own bike); set `bicycle_fee_usd` for a fixed trip fee.
- `TRANSIT`: API fare in USD, or an explicit `transit_fare_usd` override.
- `DRIVE`: requires `driving_usd_per_km`; adds `parking_usd` (default zero) and API
  toll estimates. `driving_tolls_usd` can explicitly override tolls. This is an
  operating-cost model, not a taxi/rideshare quote.

Only include modes actually available to the modeled population. All values are
one-way, per traveler. Fares, tolls, and time estimates depend on coverage and the
specified departure time. Supply a timezone-aware `departure_time` for comparisons;
by default the function fixes the current UTC time for the call. The CLI accepts
`--departure-time` with an ISO 8601 timezone. Keep beta, omega, and c0 shared across
the graph when required by your model.

The function requests alternative routes and minimizes over the returned candidates;
Google does not guarantee enumeration of all routes or minimization of your custom
monetary/time objective. Missing fares or unknown toll amounts raise an error rather
than treating them as zero or silently excluding a potentially cheapest mode.
Non-USD prices require explicit conversion outside this function. Empty route results
exclude a mode; API/network failures raise errors. If every mode has no route,
`f=0`, `cost_usd=inf`. Identical points use `f=exp(-beta*c0)` without an API call.
`log_f` preserves the log weight if the exponential underflows. CLI JSON represents
infinite cost/log weight as `null`.

For downstream normalization, use log weights (subtract each row's maximum before
exponentiating) to avoid underflow. The row-stochastic travel matrix has largest
eigenvalue 1. Your race-weighted matrix `A = P diag(race_proportions)` is instead
substochastic; its spectral radius, not that of `P`, is the proposed index.

API references:
[Compute Routes](https://developers.google.com/maps/documentation/routes/reference/rest/v2/TopLevel/computeRoutes),
[transit fares](https://developers.google.com/maps/documentation/routes/transit-route),
and [toll estimates](https://developers.google.com/maps/documentation/routes/calculate_toll_fees).

## NYC pilot: 70 zones and route matrices (collection disabled)

Open `notebooks/APMA4903_NYC_routes.ipynb` for the working notebook. The supplied
notebook is preserved byte-for-byte at `notebooks/reference/APMA4903_original.ipynb`.
The working notebook runs offline by default, including when running all cells.
No Routes API calls were made while preparing this workflow. Collection requires
explicit user approval before enabling `ALLOW_API` and a positive request cap.

### Geographic split

`scripts/build_nyc_zones.py` uses the existing local 2020 tract shapefile and
`outputs/ny_2020/tracts/nodes.csv`; it makes no downloads. It groups all **2,327 NYC
tracts into exactly 70 zones**, keeping borough boundaries and whole tracts intact.
Largest-remainder population allocation gives Bronx 12, Brooklyn 22, Manhattan 13,
Queens 19, and Staten Island 4. Within each borough, deterministic recursive spatial
bisection along the population-weighted principal axis balances population.
Demographic composition is not used in the partition.

All **8,804,190** residents and all 75 population/demographic columns are conserved.
Zone populations range from **119,233 to 135,465**, averaging 125,774. Zones are
custom analytical units, not official neighborhoods; equal population does not
imply equal area. Strict contiguity is not guaranteed. Census polygons include
water. Each routing point is the internal point of a populated tract nearest the
population-weighted mean of tract internal points (in UTM 18N metres). These points
approximate residents' origins and require review before acquisition; they are not
verified station entrances or household-weighted centroids. One point per zone also
hides within-zone travel variation. Demographics refer to 2020.

Rebuild offline from the project root:

```bash
.venv/bin/python scripts/build_nyc_zones.py
```

Outputs in `outputs/nyc_routes/`:

- `zones.csv`: ordered zone IDs, WGS84 routing points, population/demographic counts.
- `tract_to_zone.csv`: complete tract-to-zone crosswalk, preserving string GEOIDs.
- `zones.geojson` and `zones_map.png`: boundaries and routing-point preview.
- `zones_manifest.json`: method, borough allocation, input checksum, and limitations.

### Acquisition and matrix conventions

`scripts/nyc_routes.py` uses pairwise **Compute Routes**, retaining returned
alternatives and fares needed by the existing generalized-cost model. It does not
use the batch Compute Route Matrix endpoint. For 70 zones, 4,900 matrix cells minus
70 modeled diagonals means **4,830 directed requests per mode and departure time**.
The default TRANSIT-only scenario fits the approximately 4,900-request target.
The original WALK + TRANSIT model needs **9,660** requests. These are request counts,
not billing estimates; retries add requests. Different directions are not assumed
symmetric.

`collect()` defaults to a dry run. Live acquisition additionally requires
`allow_api=True`, a positive integer `max_requests`, and a locally configured
`GOOGLE_MAPS_API_KEY`. The notebook ships with `ALLOW_API=False` and
`MAX_REQUESTS=0`. A single-request pilot can use a cap of 1 after approval.
Successful responses are immediately committed to SQLite, and resuming skips them.
There are no automatic retries: errors stop the run, and failed/interrupted attempts
require `retry_errors=True` after inspection. An interrupted request may have been
billed even if no response was saved. Keys are never stored in the cache.

The proposed notebook departure is **October 20, 2026, 8 a.m. America/New_York**;
select and agree on an actual snapshot before collection. All pairs use the same
timestamp. The cache fingerprints zone definitions, ordering, modes and departure
time and rejects mismatches. Each new scenario needs its own cache and output
directory. Price/model parameters can be changed offline using saved responses.

`export_matrices()` makes no network requests. It exports labeled CSVs and
`matrices.npz` with `f`, `log_f`, `cost_usd`, `price_usd`, `duration_seconds`,
`distance_meters`, `best_mode`, `status`, zone order and population. Rows are origins,
columns destinations. Duration/distance/price refer to the returned alternative
minimizing generalized cost over the requested modes. The separate
`fastest_duration_seconds_by_mode` tensor has axes `(mode, origin, destination)` and
retains usable times even when fares are missing. All returned alternatives remain
in SQLite for future offline analysis. The export manifest records model parameters.

- `local_model`: diagonal cost `c0`, weight `exp(-beta*c0)`, zero time/distance by
  modeling convention, not measured local travel.
- `ok`: fully evaluated pair.
- `no_route`: all modes returned no route; weight 0, cost/time/distance infinity.
- `pending`, `request_error`, `invalid_response`, `price_unknown`: unresolved;
  cost and weight remain NaN and cannot be used for the index.

Missing fares are not assumed free; explicit justified fare overrides can be supplied
at export. Walking/cycling ferry candidates likewise require explicit modeling and
remain price-unknown. No-route responses and API errors are distinct.
`population_transition()` normalizes log weights against destination population and
rejects missing data. The notebook constructs `A = P @ diag(p)` and its spectral
radius using the original Black-alone definition `(NH_BLACK + H_BLACK) / TOTPOP`.
The supplied beta, omega and c0 values remain illustrative, not calibrated.

Verification uses only mocks, including blocked-network notebook execution:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

API behavior was checked against Google's
[Compute Routes documentation](https://developers.google.com/maps/documentation/routes/compute_route_directions)
and [transit route documentation](https://developers.google.com/maps/documentation/routes/transit-route),
including explicit departure timestamps, alternatives, and potentially missing fares.

### Verified 2020 race counts for the 70 zones

Run `.venv/bin/python scripts/prepare_zone_demographics.py` to recompute entirely
from local Census files. No Routes API or other network access is used. The script
independently reads official P1/P2 records and the Census geographic header, joins
by `LOGRECNO`, and aggregates the existing tract-to-zone crosswalk. It checks every
existing NYC tract and zone count, then checks tract aggregates against separately
published county records. The 70 zones and their ordering are unchanged.

`outputs/nyc_routes/demographics_2020/` contains:

- `race_counts_2020.csv`: all 70 zones, total population, and seven exhaustive race
  groups regardless of Hispanic origin (six race-alone groups and two-or-more-races).
- `race_ethnicity_counts_2020.csv`: Hispanic/Latino of any race plus seven
  non-Hispanic race categories, forming eight non-overlapping groups.
- `race_by_hispanic_origin_2020.csv`: the complete 14-category cross-tabulation per
  zone, with counts and fractions of zone population.
- `demographics_2020.npz`: matching counts/proportion arrays, population, and ordered
  zone/category names; race shape `(70,7)`, race/ethnicity `(70,8)`, and race by
  Hispanic origin `(70,2,7)`.
- `nyc_race_totals_2020.csv` and `borough_race_totals_2020.csv`: aggregate summaries.
- `data_dictionary.json`: exact P1/P2 variable mappings and category definitions.
- `validation.json`: validation results, source URLs, and input checksums.

All counts cover all ages on April 1, 2020. Race-alone counts exclude persons
reporting that race together with another; those persons are in two-or-more-races.
Hispanic origin is separate from race in this Census, so Hispanic totals overlap
with the first table and must not be added to its seven groups. Both the seven-group
race table and the eight-group race/ethnicity table sum independently to each zone's
total population. The working notebook loads these verified 2020 arrays for PSSI;
its older 2010 loader definitions remain only in the reference section.

### Exactly 70 groups, with an offline alternative

**2,327 source tracts → 70 analysis groups → 70 × 70 travel matrices.** The source
tracts are only the building blocks for aggregating demographics. They are never
passed individually to the route collector. `outputs/nyc_routes/groups_70.csv` is a
single 70-row table with group IDs, borough, constituent tract count, coordinates,
population and seven 2020 race counts. The notebook asserts that there are exactly
70 groups and that demographic and travel matrix ordering agree.

The notebook now defaults to `TRAVEL_TIME_SOURCE = "offline"`. It uses
`scripts/approximate_travel.py`, which requires no networking, credentials or
additional data. Reproduce from the project root:

```bash
.venv/bin/python -m scripts.approximate_travel
```

For distinct groups, estimated minutes are
`access_wait_minutes + 60 * detour_factor * great_circle_km / effective_speed_kmh`.
The diagonal is zero by modeling convention, with the existing cost floor `c0`.
These adjustable scenarios are **uncalibrated assumptions**, not measured travel
speeds, empirical bounds, or network routes:

| Scenario | Speed, km/h | Distance multiplier | Access/wait, minutes |
|---|---:|---:|---:|
| faster | 25 | 1.2 | 5 |
| baseline | 18 | 1.4 | 10 |
| slower | 12 | 1.6 | 15 |

Each scenario is saved under `outputs/nyc_routes/offline_approximation/<scenario>/`
with labeled `duration_minutes.csv`, distance, cost and social-weight CSVs plus
`matrices.npz`. All matrices are 70 × 70, symmetric, finite, and marked approximate.
The manifest records assumptions and zone identity. The default generalized cost
values time only (`trip_cost_usd=0`); a fare must be supplied as an explicit modeling
assumption if desired. This does not assert that actual transportation is free.

This approximation ignores street topology, bridge/tunnel constraints, ferry
schedules, transit networks, traffic and time of day. It treats all groups as
reachable and is especially weak across water and for Staten Island. Use it for
initial sensitivity analysis, not validated transit estimates. It does not estimate
within-group variation. The offline notebook path can calculate the segregation
index immediately using the verified 2020 demographic arrays. API collection remains
disabled, and API data is stored separately; selecting the API analysis source does
not itself enable collection.

### Actual subway lines and scheduled travel times

The default notebook source is now `subway_hybrid`. `scripts/subway_travel.py` uses
the official [MTA regular subway GTFS](https://www.mta.info/developers) downloaded to
`data/transit/gtfs_subway.zip`, including SIR. The ZIP includes `stops.txt`,
`shapes.txt`, `trips.txt`, `stop_times.txt`, `calendar.txt`, `calendar_dates.txt`,
and `transfers.txt`. No Google Routes calls or credentials are needed. Routing is
entirely local after this public dataset download. The downloaded feed version is
`20260826-X-long-term-supplement-trip-ids`, valid May 26–October 31, 2026.

```bash
.venv/bin/python -m scripts.subway_travel --date 2026-10-20 --departure 08:00:00
# Optional explicit public-dataset refresh, not a Google Routes request:
.venv/bin/python -m scripts.subway_travel --download --date 2026-10-20 --departure 08:00:00
```

A connection-scan router selects the earliest modeled arrival using actual stop
sequences and scheduled times. It applies weekday/calendar exceptions, handles
GTFS times above 24:00 and adjacent service days, and keeps both train directions.
Access is permitted at multiple stations. Staying aboard the same train includes
dwell time without adding a transfer penalty. Transfers follow GTFS minimum times,
with an explicit assumed 120-second floor and a 180-second default for reboarding
at the same platform. Service dates outside feed validity are rejected.

Walking uses 4.8 km/h, a 1.3 distance multiplier, a 3 km access/egress limit,
2 minutes for each station entry/exit, and a 5 km limit for direct walking.
Straight walking segments must lie within a 60 m buffered union of land-containing
Census blocks. This prevents obvious open-water shortcuts but is not pedestrian
street routing; valid bridge approaches can be omitted. There are no synthetic
walking transfers between unrelated nearby stations.

The saved 8 a.m. October 20 snapshot has 70 × 70 matrices in
`outputs/nyc_routes/subway_network/2026-10-20_080000/`:

- `subway_only/duration_minutes.csv`: 3,494 nonlocal pairs by scheduled rail and
  70 by approximate direct walking; uncovered entries are infinity. Missing paths
  are a limitation of this model, not proof that travel is impossible.
- `hybrid/duration_minutes.csv`: complete matrix with the remaining 1,266 pairs
  filled by the earlier distance-only baseline. `best_mode.csv` labels every
  `DISTANCE_FALLBACK`; it must not be represented as a subway prediction.
- Both directories contain `matrices.npz`, statuses, coverage, and time-only
  generalized costs/social weights. Subway-only missing costs/weights are NaN;
  hybrid fallback component breakdowns are NaN because no rail itinerary exists.
- `journeys.jsonl`: access, waiting, train legs, transfer and egress details for
  modeled rail trips. Scheduled times are seconds from local service-day midnight.
- `subway_lines.geojson`, `stations.csv`, `routes.csv`, and
  `subway_groups_map.png`: actual feed line shapes/stations and the 70 groups.
- `coverage_by_group.csv` and `manifest.json`: access counts, source checksum,
  feed validity, assumptions and coverage. `distance_meters` is NaN because a full
  path distance is not computed; great-circle group distance is saved separately.

This combines **2020 Census demographics with the stated 2026 service schedule**.
The regular MTA feed is a normal schedule, not a reconstruction of 2020 service,
real-time conditions, or all temporary service changes. Buses, ferries and commuter
rail are omitted; Staten Island therefore has no modeled rail link to the other
boroughs. Trips are limited to a six-hour arrival window. All economic weights are
time-only and illustrative; no fare is assumed. Select `subway_only` to avoid any
fallback (full-city index calculation then stops on missing pairs), `offline` for
the distance scenarios, or `api` for the separately collected Google dataset.
Selecting a data source never enables Google collection.

### Final two-mode generalized cost and segregation matrices

The final analysis now uses `scripts/cost_matrices.py`, superseding the earlier
time-only weights. It uses only WALK and TRANSIT and applies the user's definition:

`C_ij = max(c0, min(omega * walk_hours, transit_price + omega * transit_hours))`

`F_ij = exp(-beta * C_ij)`

`A_r[i,j] = F[i,j] * D[j] * p[j,r] / sum_k(F[i,k] * D[k])`

Both mode candidates are retained even if one is slower; selection minimizes
**dollars plus valued time**, not time alone. Scheduled transit candidates are
kept separately for zero and standard-fare journeys before minimizing. WALK is
zero out of pocket. Paid transit uses the standard adult $3 fare verified against
[MTA fares](https://www.mta.info/fares-tolls/subway-bus) and its
[2026 changes](https://www.mta.info/fares-tolls/2025-changes). SIR-only journeys
without either endpoint at St George or Tompkinsville are free, following the
[MTA tariff](https://www.mta.info/document/20581). Normal transfers are treated as
one fare; passes, caps, discounts and fare history are not modeled. Earliest-arrival
labels do not enumerate every possible fare-dependent itinerary.

Default beta=0.1 per dollar, omega=$15/hour, c0=$1.25 remain illustrative. The same
c0 is used everywhere, including `C[i,i]=c0`; the cost floor is applied after the
mode minimum. The diagonal remains in the population-normalization denominator.

```bash
.venv/bin/python -m scripts.cost_matrices
```

All inputs are local. The first run recomputes independent mode candidates from
the saved MTA ZIP; later runs reuse `outputs/nyc_routes/cost_model/mode_inputs.npz`.
This allows beta, omega, c0 and standard fare changes without API calls. Recompute
mode inputs explicitly if zones/feed change. The `mode_inputs_manifest.json`
records source identities and constraints.

The 70 × 70 outputs are in `outputs/nyc_routes/cost_model/`:

- `estimated/C_usd.csv`, `F.csv`, `log_F.csv`: final generalized cost and social weights.
- `estimated/chosen_mode.csv`: WALK, TRANSIT or LOCAL. `chosen_source.csv` distinguishes
  approximate walking, scheduled transit, and a hypothetical transit proxy.
- Independent times, prices and generalized costs for both modes are exported,
  alongside `estimated/cost_matrices.npz` and `zone_order.csv`.
- `estimated/A_<race>.csv` and `race_analysis.npz`: seven race-specific matrices,
  population-normalized transition matrix, proportions and ordered IDs.
- `estimated/race_indices.csv`: Perron eigenvalues and eigenpair residuals.
- `estimated/race_section_scores.csv`: right eigenvectors, each normalized to sum 1.
- Corresponding `race_ethnicity_*` files contain the separate eight-category
  mutually exclusive Hispanic-origin/race breakdown.

Direct WALK no longer has an arbitrary 5 km cap but retains the conservative land
mask. The scheduled transit model still has 3 km access/egress and a six-hour
arrival horizon. `estimated` adds a **hypothetical TRANSIT proxy** only where no
scheduled transit candidate exists: 18 km/h, distance multiplier 1.4, 10 minutes
overhead, and an assumed $3 fare. It is not a third mode or a quoted real itinerary.
Default selections are 181 WALK, 3,365 scheduled TRANSIT, 1,284 proxy TRANSIT and
70 local entries. This coverage differs from the earlier time-only hybrid because
fares and independent mode candidates are now included.

`scheduled_candidates_only` omits the proxy; 894 costs remain NaN, and a full-city
index is not computed. Missing modeled paths do not prove real-world
inaccessibility. The complete estimated analysis remains sensitive to proxy
assumptions, omitted buses/ferries, approximate walking and representative points.
It uses 2020 demographics with the stated 2026 schedules/fares.

`race_analysis` uses stable log-space normalization, preserves directional weights,
checks the literal A formula, and validates `A @ v = lambda * v`, nonnegative v,
`sum(v)=1`, and eigenvalues in [0,1]. These normalized right-eigenvector section
scores are relative model weights, not individual segregation probabilities.
The notebook defaults to the final cost model; Google collection remains disabled.

### Four-category comparison, Fryer SSI and population-share adjustment

The main notebook comparison now uses **White alone, Black alone, Hispanic/Latino
(any race), and Asian alone**, in that order. White, Black and Asian alone include
Hispanic residents. These four groups overlap and do not partition NYC; each
category's share uses all 8,804,190 residents in the verified 2020 Census counts.
The original seven-race and eight-race/ethnicity exports remain available.

Run `.venv/bin/python -m scripts.cost_matrices` to regenerate everything from
saved local inputs. The primary files under `outputs/nyc_routes/cost_model/estimated/`
are now:

- `four_category_indices.csv`: population counts, citywide shares, PSSI and Fryer
  scores, separate subtraction and normalization columns, majority-section counts,
  isolation and dissimilarity.
- `four_category_comparison.html`: scrollable 19-column results table plus a
  complete dictionary describing every column, formula and unit.
- `four_category_column_dictionary.csv`: the same definitions in tabular form.
- `four_category_analysis.npz`: four 70 × 70 A matrices, common P, counts,
  proportions, eigenvalues, right eigenvectors, category names and zone order.
- `four_category_section_scores.csv`: 70 rows of right eigenvector scores.
- `four_category_section_diagnostics.csv`: 280 category/section rows containing
  population, group counts/shares, majority flags, network same-group exposure and
  right eigenvector scores. The eigenvectors each sum to one.

For each category, `K = sum_i(D_i * p_ir) / sum_i D_i`, and the requested
normalization is **`(score - K) / (1 - K)`**. K is a fraction in [0,1], not an
unweighted average of zone shares or a percentage from 0 to 100. Negative results
are retained; K=1 is undefined. This is a population-share adjustment, not a
simulated chance correction. It is applied to the scalar PSSI and isolation
scores, not to individual eigenvector components. The requested **adjusted Fryer**
score is separately **`SSI - K`**, with no division. The previously calculated
`(SSI - K)/(1 - K)` remains available as an additional **rescaled Fryer** column.
PSSI also has an unscaled `lambda - K` column to make the distinction explicit.

`scripts/segregation_metrics.py` implements the original notebook's **majority-section
adaptation of Echenique–Fryer SSI**: select sections with `p_ir >= 0.5`, take their
induced submatrix of `P_ij = F_ij * D_j / sum_k(F_ik * D_k)`, and average component
spectral radii weighted by each component's number of selected sections. This uses
the same population-weighted P as PSSI, retaining local diagonal weights and
directional intensities. It does **not** renormalize induced rows; otherwise it
would discard out-group interactions. Support must be reciprocal. When there are
no majority sections, the original notebook's zero convention is used and flagged.
This zero does not imply that the category's residents experience no segregation.

The [original paper](https://eml.berkeley.edu/~fechenique/published/echenique-fryer-qje.pdf)
uses majority-classified Census blocks and one-kilometer neighbors in its residential
application. Our much larger 70 sections and modeled travel interactions define a
different empirical network; these scores are not an individual-level estimate or
a replication of that application. As requested, even the Fryer normalization uses
population share K, rather than the fraction of majority sections. On pp. 24–25,
the paper defines adjusted SSI by subtracting a simulated random-assignment
baseline, which is not generally K. We have not computed that simulation baseline;
our `SSI - K` column is labeled as a population-share adjustment.

Additional computations, with `N_r = sum_i D_i*p_ir` and `N = sum_i D_i`:

- Residential isolation: `sum_i (D_i*p_ir/N_r) * p_ir`.
- Network isolation: `sum_i (D_i*p_ir/N_r) * (P @ p_r)_i`, the modeled same-group
  exposure averaged over the category's residents.
- Dissimilarity versus all other residents:
  `0.5 * sum_i abs(D_i*p_ir/N_r - D_i*(1-p_ir)/(N-N_r))`.
  This retains its usual 0–1 scale without an additional K adjustment.

With the saved defaults (beta=0.1, omega=$15/hour, c0=$1.25, standard fare=$3):

The exact current table, including the new `PSSI - K` and `Fryer SSI - K` columns,
is regenerated in `four_category_comparison.html` and `four_category_indices.csv`.
Every CSV column is defined in `four_category_column_dictionary.csv` and displayed
below the HTML table and in the notebook.

There are respectively 18, 10, 11 and 1 majority sections. The negative adjusted
Asian Fryer result follows the requested formula and its majority-section score
being below population share K; it should not be interpreted as a negative
probability or a universal finding of integration. All values remain conditional
on the modeled travel network, transit proxies, and illustrative cost parameters.
The cost matrix and 70 sections are unchanged. No Routes API calls are needed.

## Eleven-city tract comparison

`notebooks/APMA4903_multi_city.ipynb` extends the four-category study to Philadelphia,
Miami, Los Angeles, Salt Lake City, San Francisco, Washington DC, Houston, Detroit,
Chicago, Phoenix and Atlanta. These results use **2020 Census city limits and
individual populated tract portions**, as selected by the user. They are not
metropolitan areas and are not aggregated to 70 groups.

`scripts/multicity_census.py` joins the official 2020 P.L. 94-171 geographic header
and P1/P2 segment by LOGRECNO, selects city-assigned blocks by PLACE, and sums them
into tract portions. Every city's total population, White alone, Black alone,
Hispanic any race and Asian alone counts are independently checked against the
official place-level records. No counts are interpolated by area. Census tract
geometries are clipped to the city polygon. Representative points are populated
block internal points nearest the population-weighted centroid; the unsnapped
centroid is also retained. The data include 4,738 populated tract portions across
the 11 cities; zero-population portions remain in validation outputs.

Houston has 170 tract portions below 100 residents, totaling 3,472 people. Keeping
these preserves city totals but affects tract-count-based scores: they constitute
18.0% of Houston's nodes and 0.151% of its residents. All cities export flags and
full-tract/city-portion population shares. PSSI uses population-weighted destination
interactions, while the Fryer component average weights majority tracts equally.

`scripts/multicity_sources.py` records 22 public GTFS feeds covering rail and bus
in every city. Most are primary agency/vendor downloads. WMATA and Phoenix are
explicitly identified public MobilityData archives of agency-produced files.
The full manifest lists included and omitted agencies, URLs, checksums, calendar
validity and timezones. This is not exhaustive coverage of every regional operator.
Ten cities use October 20, 2026; San Francisco uses the common valid Tuesday
August 25, 2026 because its saved Muni feed ends August 28. Feed dates are never
extended silently. Dataset attribution is reproduced in the report and
`outputs/multicity/data_attribution.txt`.

`scripts/multicity_transit.py` uses directed GTFS stop-sequence patterns with median
scheduled ride/dwell times and expected boarding waits (half mean headway in the
07:00–10:00 local service window). It honors calendars, exceptions, frequencies and
boarding/alighting restrictions. Conservative transfer buffers handle restrictive
transfer rules; interlining and timetable synchronization are not modeled. This is
an expected-time graph, not an exact departure-time itinerary or real-time prediction.

Walking is great-circle distance × 1.3 at 4.8 km/h; access limits are 1 km for bus
and 3 km for rail, with 400 m pedestrian transfers and documented entry/transfer
buffers. County Census water polygons screen direct chords used for transit access,
egress, transfers and direct walking. These checks may reject legitimate bridge or
indirect land routes and are not pedestrian street routing. For citywide indices,
only pairs lacking **both** a screened walking candidate and a GTFS transit candidate
receive a flagged distance-only walking completion. Such completions can cross
water; per-city fallback counts are part of the comparison. The strict screened
cost matrix is also exported with infinity for unresolved candidates. Missing
modeled paths do not prove a real journey is impossible.

All cities use beta=0.1 per dollar, omega=$15/hour, c0=$1.25, and a **standardized
illustrative $3 per transit journey** including transfers. The fare is a shared
comparison scenario, not a quote of each agency's fares. C, F, P and A follow the
same formulas as the NYC model. Only WALK and public TRANSIT enter the cost minimum.

The primary outputs under `outputs/multicity/` are:

- `city_comparison.html` / `.csv`: 44 city/category results, raw and adjusted PSSI
  and Fryer, isolation, dissimilarity, category population shares and majority counts.
- `city_coverage.csv`: tract counts, service dates, bus/rail route counts, modeled
  transit coverage, fallback counts and small-tract diagnostics.
- `score_column_dictionary.csv` and `tract_score_dictionary.csv`: formulas,
  meanings and units for every comparison and tract-level column.
- `<city>/analysis/tract_scores.csv`: all tract/category eigenvector and local SSI
  information, including repeated city eigenvalues and Fryer component eigenvalues.
- `<city>/analysis/analysis_matrices.npz`: ordered P, four A matrices, eigenvalues,
  right eigenvectors, counts, proportions and tract GEOIDs (load GEOIDs as text).
- `<city>/analysis/cost_matrices.npz`: C, F, chosen mode/source and fallback flags.
  `barrier_screened_costs.npz` retains the incomplete alternative.
- `<city>/travel/modes.npz`: walking/transit duration arrays, barrier flags and
  station access counts. `manifest.json` records the model, dates and diagnostics.
- `<city>/demographics/`: verified counts, clipped tract polygons, water polygons,
  zero-population portions and source/checksum/validation records.

There is **one PSSI eigenvalue per city/category**, not a separate eigenvalue per
tract. Its sum-one right eigenvector provides tract scores; `n*v_i` is also exported
on a mean-one scale across cities with different tract counts. For Fryer, each
majority component has an eigenpair `(lambda_C, v_C)` with `sum(v_C)=1`, and the
tract SSI is `|C| * lambda_C * v_C[i]`. Averaging these over majority tracts recovers
the city score. Nonmajority tracts have undefined local Fryer scores, exported
blank. A city/category with no majority tracts retains the original notebook's
zero aggregate convention and an explicit status flag; this does not establish
that its residents experience no segregation.

PSSI normalization remains `(lambda-K)/(1-K)`; requested adjusted Fryer is `SSI-K`.
The optional rescaled Fryer column is `(SSI-K)/(1-K)`. K uses city population counts,
not the fraction of majority tracts. This does not reproduce the original paper's
simulated null baseline. Local eigenvector components are not normalized with K.

After source acquisition, all routing, matrices and scores regenerate locally:

```bash
.venv/bin/python -m scripts.run_multicity
# Optional cached subset:
.venv/bin/python -m scripts.run_multicity --cities philadelphia miami
.venv/bin/python -m unittest discover -s tests -q
```

The multicity pipeline contains no Google Routes client. Source acquisition is
separate from local computation; no Routes API calls are authorized or made.

## Transit available versus walking only

`notebooks/APMA4903_transit_comparison.ipynb` compares the same 11 cities, 4 racial
categories and 4,738 populated city-limit tract portions in two scenarios:
walking plus the saved rail/bus candidates, and walking only. The 88 results and
37,904 tract/category/scenario rows are under `outputs/multicity/transit_comparison/`.

Both scenarios use **identical complete distance-based walking estimates**. This
is necessary because the previous report added walking completions only where
transit was unavailable; turning off transit under that rule would also change
the walking network. The paired with-transit scores are therefore recomputed.
The earlier report is preserved, and `previous_report_reference.csv` quantifies
the differences. Walking distance proxies can cross water and do not represent
validated pedestrian itineraries. Water flags and their population-weighted
transition mass are exported in `walking_assumptions.csv`.

Shared demographic, geographic and economic inputs remain unchanged. Each pair
uses walking distance × 1.3 at 4.8 km/h, omega=$15/hour, beta=0.1 per dollar,
c0=$1.25 and a standardized illustrative $3 fare when transit is selected.
The walking-only scenario sets all transit candidates to infinity. Tests verify
that removing transit cannot lower any generalized cost. That cost inequality
does not itself guarantee the direction of a segregation-score change because
the interaction matrix is row-normalized.

The report `index.html` includes four paired heatmaps with matching scales:
PSSI (`lambda`), normalized PSSI (`(lambda-K)/(1-K)`), adjusted Fryer (`SSI-K`),
and raw Fryer SSI. `all_four_metrics.png` combines all eight panels.
No-majority-tract Fryer cases have an asterisk: raw SSI=0 by the existing
convention, adjusted SSI=-K. These cases do not establish an absence of segregation.
Small isolated tract portions can dominate a spectral score; population flags
and local eigenvectors remain available for inspection.

`scenario_scores.csv` contains all scenario/category results;
`paired_comparison.csv` includes each metric with transit, walking only, and the
signed change **walking only minus with transit**. The three column dictionaries
define scenario-level, paired, and tract-level output columns. Each city/scenario
subfolder includes the complete C, F, P and A matrices, eigenpairs and tract scores.

`scripts/verify_transit_comparison.py` independently reconstructs P and A, verifies
the common walking inputs and normalization formulas, checks eigenpair residuals
and Fryer local/global aggregation, and writes `verification.json`. Nearly
disconnected walking-only matrices can have slow power-iteration convergence;
a dense eigensolver fallback must pass the same Perron-pair checks before export.

Reproduce the comparison using saved arrays, with no network access required:

```bash
MPLCONFIGDIR=/private/tmp/apma4903-matplotlib .venv/bin/python -m scripts.transit_comparison
.venv/bin/python -m scripts.verify_transit_comparison
```

The initial paired run was executed with socket connections disabled. This
comparison holds residential populations fixed and is not a causal prediction
of abolishing a transit system.

## Distance-based Fryer SSI and NYC eigenvector maps

`notebooks/APMA4903_distance_fryer_and_maps.ipynb` adds the residential distance
network requested separately from the earlier transit-weighted majority-tract
adaptation. `outputs/distance_fryer/index.html` contains the results, definitions,
plots, sensitivity checks and links to the Houston/Phoenix diagnosis and NYC maps.

The main Fryer model covers **170,059 populated Census blocks in 12 cities**: the
11 comparison cities plus NYC. A companion model uses 7,030 populated city-limit
tract portions. All counts are 2020 Census; block and tract totals agree exactly.
NYC block counts were extracted from the already-downloaded P.L. archive and
checked against the official city record. The default neighbor radius is 1 km,
with 0.5 km and 1.5 km sensitivity results: 288 city/category/scale/radius rows.

The implementation follows the paper's distance network and checks the authors'
published MATLAB code/data (`data/references/fryer/citations.json`):

- Distinct nodes within the great-circle cutoff are neighbors. No self-links.
  Earth radius is 6378.1 km, as in the authors' code. R_ij=1/d_i, where d_i counts
  all neighbors regardless of category. No population or transit weights enter R.
- Each node receives the largest of the four category counts, not necessarily a
  strict >50% majority. This matches the dominant-category labels in the authors'
  published sample. The explicit tie order is Hispanic, Asian, Black, White;
  original preprocessing code is unavailable. All-four-zero nodes remain other.
  Resident category counts overlap, but assigned node labels do not.
- Restrict R to the category without renormalizing. As in `neighbors.m`, exclude
  nodes with no neighbors from the SSI denominator. Keep assigned nodes with
  only other-category neighbors as zero-score singletons. No eligible assigned
  nodes means undefined, rather than the older adaptation's zero convention.
- Average component Perron eigenvalues by eligible component node counts.
  Paper-scaled local SSI is component size × eigenvalue × sum-one right vector.
  Its mean equals city SSI; local scores may exceed 1. Components include
  singleton zero-score observations, documented separately from edge-bearing groups.
- K is the category fraction of **all city residents**, independent of node
  labels and isolate exclusions. Export raw SSI, requested SSI-K, and
  (SSI-K)/(1-K). The paper's simulated-baseline correction is not computed.

`block_scores_1km.csv` gives 48 main results; `tract_scores_1km.csv` gives the
48 tract companions. `pssi_and_distance_fryer.csv` compares these with the existing
11-city PSSI scenarios, keeping both node scales explicit. NYC's existing PSSI
uses 70 sections and remains separate. Each city/scale directory stores ordered
nodes, sparse adjacency/R/category-B matrices, B index maps, component eigenpairs
and `local_scores_1km.csv.gz`. Three column dictionaries explain summary, local
and side-by-side fields. The radius-sensitivity table includes all 288 results.

Interpretation limits: dominant labels discard within-node population mixtures;
equal block weights ignore population density; city limits omit outside neighbors;
straight-line links can cross water. A zero SSI for an assigned-node network does
not establish absence of segregation among all category residents. The geographic
units, year, boundary choice and explicitly documented assignment rule make this
a new application, not a replication of the original 2000 MSA estimates.

`scripts/diagnose_transit_similarity.py` explains Houston and Phoenix using saved
PSSI matrices and GTFS stop-access inputs. Transit lowers generalized costs for
44.91% and 64.52% of ordered location pairs respectively. Near-equal White/Asian
PSSI values are driven by unchanged subnetworks with zero modeled boarding access.
Most such exclusions result from the configured bus1km/rail3km access distances,
after the1.3 walking multiplier. Houston's influential Kingwood/Cypress locations
are near park-and-ride stops outside that cutoff; Phoenix's strongest White driver
is far from active stops. These are model-access limitations, not proof that real
transit is impossible. Black/Hispanic PSSI changes substantially. The diagnostic
report and CSVs are in `outputs/multicity/transit_diagnostics/`.

`scripts/plot_nyc_eigenvectors.py` makes four NYC maps from the existing verified
70-section PSSI eigenvectors. `outputs/nyc_routes/eigenvector_maps/index.html`
contains separate maps, a combined 2×2 figure and exact section CSVs. Colors show
70*v_i (section mean1) on a shared logarithmic scale. They are relative section
scores, not citywide eigenvalues or population-share normalized scalar PSSI.
The saved NYC model includes subway/SIR plus its documented transit proxies;
it does not add an actual bus or ferry network. Only cartographic display geometry
was clipped to saved land-containing blocks; analytical IDs/counts/scores are intact.

Reproduce using local saved data:

```bash
MPLCONFIGDIR=/private/tmp/apma4903-matplotlib .venv/bin/python -m scripts.distance_fryer
.venv/bin/python -m scripts.verify_distance_fryer
.venv/bin/python -m scripts.diagnose_transit_similarity
MPLCONFIGDIR=/private/tmp/apma4903-matplotlib .venv/bin/python -m scripts.plot_nyc_eigenvectors
```

The Fryer build was executed with socket connections disabled. The verification
checks defining neighbor weights, exact category-B restrictions, assignments,
isolate treatment, eigenpair residuals, local/global aggregation, and both requested
adjustments. NYC map validation also checks every saved A*v=lambda*v and section ID.

## Revised walking access to transit and maps for all cities

Open [the offline review notebook](notebooks/APMA4903_walk_access_revision.ipynb)
or `outputs/multicity/walk_access_revision/index.html`. The revision covers
12 cities: the 11 comparison cities retain individual populated 2020 city-limit
tract portions, while NYC retains its 70 custom sections. These spatial scales
remain explicit. White alone, Black alone, Hispanic of any race, and Asian alone
are analyzed separately; Hispanic overlaps the race-alone categories.

The initial bus/rail access and final egress **distance caps are removed**.
Walking to an actual stop, riding GTFS service, and walking from the final stop
all contribute their full time, including entrance/exit buffers. A transit
candidate requires at least one actual ride. Conservative water/land chord checks
remain and can exclude valid bridges or street detours. Automatic transfer links
within 400 m, explicit GTFS links and restrictions, and chains of walking links
are retained: **400 m is not a universal cap on total transfer walking**.

Both revised scenarios use the same direct walking proxy: great-circle distance
times 1.3 at 4.8 km/h, including flagged potential water crossings. Shared
parameters are beta=0.1 per dollar, omega=$15/hour, c0=$1.25, and an illustrative
standardized $3 transit fare. Cached service dates remain October 20, 2026, except
San Francisco, August 25, 2026. NYC uses actual cached subway/SIR only; it has no
bus/ferry/commuter-rail network or older hypothetical transit replacements. NYC's
change from its previous router is therefore not an isolated access-cap experiment.

Raw PSSI is the city/category eigenvalue lambda; normalized PSSI is
`(lambda-K)/(1-K)`, using the category's citywide resident fraction K. Each right
eigenvector sums to 1. The four maps for every city show `n*v_i`, whose mean over
nodes is 1, on a shared logarithmic color range of 0.25–4. Exact values and values
outside that displayed range are retained and documented. Map values are neither
city eigenvalues nor scalar population-share normalizations.

The **separate distance-based Fryer model** is recomputed on 1 km block-neighbor
graphs with a tract companion. It uses dominant-category labels and geographic
neighbors, not transit or PSSI's population-weighted interactions. Raw SSI,
requested `SSI-K`, and `(SSI-K)/(1-K)` are included; none depends on the transit
scenario. The older travel-weighted majority adaptation appears only in explicitly
named legacy diagnostic files. Full geographic-method details and sensitivity
remain in `outputs/distance_fryer/`.

Key revised outputs:

- `city_scores.csv`, `scenario_scores.csv`, and `distance_fryer_recomputed.csv`:
  PSSI comparisons and separately identified block/tract Fryer scores.
- `all_node_scores.csv` and the column dictionaries: exact node eigenvectors,
  identifiers, scales, and metric definitions.
- `<city>/<scenario>/`: ordered C/F/P/A matrices, eigenpairs, node scores, and
  manifests; `<city>/travel/` includes chosen access/egress legs and stop indices.
- `coverage.csv`, `previous_results_comparison.csv`, and `manifest.json`:
  access diagnostics, comparisons with earlier outputs, and modeling assumptions.
- `maps/index.html` and `maps/<city>/index.html`: the complete gallery and
  city pages, each with four individual PNGs, a combined four-map figure,
  `section_eigenvectors.csv`, and map metadata.
- `verification.json`: produced by the independent verifier once the outputs
  and maps are complete. Its absence does not indicate a passed verification.

Rebuild explicitly from saved local inputs; the review notebook never executes
these commands automatically:

```bash
.venv/bin/python -m scripts.walk_access_revision
# Scores only, using existing revised travel matrices:
.venv/bin/python -m scripts.walk_access_revision --analysis-only
# Aggregate tables/plots only, after all city scenarios exist:
.venv/bin/python -m scripts.walk_access_revision --finalize-only
.venv/bin/python scripts/plot_city_eigenvectors.py
.venv/bin/python -m scripts.verify_walk_access_revision --require-maps
```

All earlier result folders are preserved. This workflow uses cached sources and
does not call the Routes API or download fresh schedules. The review notebook
loads available saved results, displays missing files clearly, and does not
silently substitute earlier scores while a build is incomplete.
