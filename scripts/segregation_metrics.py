"""Local demographic selection and comparisons on a shared interaction network.

Fryer SSI here is the notebook's majority-section adaptation, evaluated on P.
It is not an individual-level estimate or a reproduction of the paper's
one-kilometer, equally weighted block-neighbor network.
"""
from html import escape

import networkx as nx
import numpy as np
import pandas as pd


FOUR_CATEGORIES = (
    'white_alone', 'black_alone', 'hispanic_latino_any_race', 'asian_alone',
)


def select_four_categories(demographics):
    """Select counts without partitioning or renormalizing overlapping groups."""
    columns = []
    for name in FOUR_CATEGORIES:
        prefix = 'race_ethnicity' if name == 'hispanic_latino_any_race' else 'race'
        names = demographics[f'{prefix}_names'].tolist()
        columns.append(demographics[f'{prefix}_counts'][:, names.index(name)])
    counts = np.column_stack(columns)
    population = np.asarray(demographics['population'], float)
    if (population <= 0).any() or (counts < 0).any() or (counts > population[:, None]).any():
        raise ValueError('Each category must be a subset of a positive zone population')
    return np.array(FOUR_CATEGORIES), counts, counts / population[:, None]


def normalize_score(score, share):
    """User's (score-K)/(1-K); retain negatives; undefined at K=1."""
    if not np.isfinite(share) or not 0 <= share <= 1:
        raise ValueError('Population share K must be a fraction in [0,1]')
    return (float(score) - share) / (1 - share) if share < 1 else np.nan


def fryer_majority_ssi(transition, proportions, threshold=.5):
    """Return SSI, number of majority zones, and number of their components.

    Keep P's original weights after restricting to majority zones. Average each
    component's spectral radius with weight (# zones in component)/(# selected
    zones), matching the notebook's convention. Retain local diagonal weights.
    No-majority-zone SSI is zero by the original notebook convention; this does
    not mean the group's people experience no segregation.
    """
    P, p = np.asarray(transition, float), np.asarray(proportions, float)
    if P.shape != (len(p), len(p)) or p.ndim != 1:
        raise ValueError('Need a square transition and one proportion per zone')
    if not np.isfinite(P).all() or (P < 0).any() or not np.allclose(P.sum(axis=1), 1):
        raise ValueError('Transition must be finite, nonnegative and row-stochastic')
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any() or not 0 < threshold <= 1:
        raise ValueError('Invalid proportions or threshold')
    # Echenique-Fryer permits directional intensities but reciprocal support.
    if not np.array_equal(P > 0, P.T > 0):
        raise ValueError('Fryer component construction requires reciprocal support')
    selected = np.flatnonzero(p >= threshold)
    if not len(selected):
        return 0., 0, 0
    induced = P[np.ix_(selected, selected)]
    graph = nx.from_numpy_array(induced > 0)
    components = list(nx.connected_components(graph))
    score = 0.
    for component in components:
        indices = sorted(component)
        radius = float(np.max(np.abs(np.linalg.eigvals(induced[np.ix_(indices, indices)]))))
        score += radius * len(indices) / len(selected)
    return score, len(selected), len(components)


def segregation_summary(P, population, proportions, names, eigenvalues, residuals, *, fryer_results=None):
    """Scalar PSSI/Fryer comparisons plus isolation and dissimilarity.

    Network isolation averages (P @ p)_i over the group's people. Residential
    isolation averages p_i over those same people. Dissimilarity compares each
    category with its complement, even when categories overlap one another.
    """
    D, proportions = np.asarray(population, float), np.asarray(proportions, float)
    if not np.isfinite(D).all() or (D <= 0).any():
        raise ValueError('Need finite positive zone populations')
    if proportions.shape != (len(D), len(names)) or len(eigenvalues) != len(names) or len(residuals) != len(names):
        raise ValueError('Category array dimensions differ')
    rows = []
    for r, name in enumerate(names):
        p = proportions[:, r]
        counts = D * p
        total = float(counts.sum())
        K = total / float(D.sum())
        fryer, majority_zones, components = (fryer_majority_ssi(P, p) if fryer_results is None else fryer_results[r])
        resident_weights = counts / total if total > 0 else np.full(len(D), np.nan)
        residential = float(resident_weights @ p)
        network = float(resident_weights @ (P @ p))
        rest = D - counts
        dissimilarity = float(.5 * np.abs(resident_weights - rest / rest.sum()).sum()) if 0 < K < 1 else np.nan
        rows.append({
            'group': str(name), 'population': int(round(total)),
            'population_share_K': K, 'population_percent': 100 * K,
            'segregation_index': float(eigenvalues[r]),
            'segregation_index_minus_K': float(eigenvalues[r]) - K,
            'segregation_index_normalized': normalize_score(eigenvalues[r], K),
            'fryer_majority_ssi': fryer,
            'fryer_majority_ssi_minus_K': fryer - K,
            'fryer_majority_ssi_normalized': normalize_score(fryer, K),
            'majority_zone_count': majority_zones, 'majority_component_count': components,
            'fryer_status': 'ok' if majority_zones else 'no_majority_zones_zero_by_convention',
            'residential_isolation': residential,
            'residential_isolation_normalized': normalize_score(residential, K),
            'network_isolation': network,
            'network_isolation_normalized': normalize_score(network, K),
            'dissimilarity_vs_rest': dissimilarity,
            'eigenpair_max_abs_residual': float(residuals[r]),
        })
    return pd.DataFrame(rows)


def comparison_column_dictionary():
    """Display labels and definitions shared by the notebook and HTML report."""
    entries = [
        ('group', 'Category', '2020 Census category. White, Black and Asian alone include Hispanic residents; Hispanic is any race. Categories overlap.'),
        ('population', 'Group population Nᵣ', 'Number of residents in this category across all 70 sections: Nᵣ = Σᵢ Dᵢ pᵢ. Unit: people.'),
        ('population_share_K', 'Population share K', 'Nᵣ / N, where N is the total NYC population across all sections. Fraction from 0 to 1; used in every adjustment.'),
        ('population_percent', 'Population (%)', '100 × K. Display only: never use this 0–100 value in the normalization formula.'),
        ('segregation_index', 'PSSI λ', 'Largest eigenvalue of Aᵣ = P diag(pᵣ), with Pᵢⱼ = Fᵢⱼ Dⱼ / Σₖ Fᵢₖ Dₖ. Raw proportion-based spectral score.'),
        ('segregation_index_minus_K', 'PSSI excess λ − K', 'Unscaled difference between PSSI and citywide population share. Zero means λ = K; negative values are retained.'),
        ('segregation_index_normalized', 'Normalized PSSI (λ − K)/(1 − K)', 'Requested normalization. Subtract K, then divide the entire difference by 1 − K. Undefined at K = 1; negative values are retained.'),
        ('fryer_majority_ssi', 'Fryer SSI S', 'Notebook adaptation: select pᵢ ≥ 0.5 sections, restrict the same P, and average component spectral radii by component section counts. Retain diagonal weights; do not renormalize restricted rows.'),
        ('fryer_majority_ssi_minus_K', 'Adjusted Fryer S − K', 'Requested population-share subtraction, without division by 1 − K. This is not the paper’s simulated-baseline adjustment S − E₀[S].'),
        ('fryer_majority_ssi_normalized', 'Rescaled Fryer (S − K)/(1 − K)', 'Additional comparison retained from the previous output. This includes division by 1 − K and must not be confused with the adjusted Fryer S − K column. Undefined at K = 1.'),
        ('majority_zone_count', 'Majority sections', 'Number of the 70 sections with at least 50% of residents in this category. This is a section count, not the category’s population share.'),
        ('majority_component_count', 'Majority components', 'Number of connected components among the majority sections using positive P support. Reciprocal support is required; directional weights may differ.'),
        ('fryer_status', 'Fryer status', 'ok means at least one majority section. no_majority_zones_zero_by_convention means the raw Fryer score is set to zero, as in the original notebook; it does not establish absence of segregation.'),
        ('residential_isolation', 'Residential isolation I', 'Σᵢ (Dᵢ pᵢ / Nᵣ) pᵢ: mean same-category residential share experienced by a resident of this category. Undefined when Nᵣ = 0.'),
        ('residential_isolation_normalized', 'Normalized residential isolation', '(I − K)/(1 − K). Population-share normalization of residential isolation; undefined at K = 1.'),
        ('network_isolation', 'Network isolation J', 'Σᵢ (Dᵢ pᵢ / Nᵣ) (P pᵣ)ᵢ: mean same-category exposure under the modeled travel interaction network. Undefined when Nᵣ = 0.'),
        ('network_isolation_normalized', 'Normalized network isolation', '(J − K)/(1 − K). Population-share normalization of network isolation; undefined at K = 1.'),
        ('dissimilarity_vs_rest', 'Dissimilarity vs. rest', '½ Σᵢ |Dᵢ pᵢ / Nᵣ − Dᵢ (1 − pᵢ)/(N − Nᵣ)|. Compares the distribution of this category with all other residents. Usual 0–1 scale, without a K adjustment; undefined at K = 0 or 1.'),
        ('eigenpair_max_abs_residual', 'Eigenpair residual', 'maxᵢ |(Aᵣ vᵣ − λᵣ vᵣ)ᵢ|. Numerical verification error, not a segregation score. Smaller is better; right eigenvectors separately sum to one.'),
    ]
    return pd.DataFrame(entries, columns=['column', 'label', 'definition'])


def write_comparison_report(comparison, folder, *, parameters, total_population, zone_count):
    """Write a standalone wide HTML table and an exact per-column dictionary."""
    dictionary = comparison_column_dictionary()
    if comparison.columns.tolist() != dictionary.column.tolist():
        raise ValueError('Comparison columns and documentation differ')
    dictionary.to_csv(folder/'four_category_column_dictionary.csv', index=False)
    labels = dict(zip(dictionary.column, dictionary.label))
    table = comparison.rename(columns=labels).to_html(index=False, border=0, float_format=lambda x: f'{x:.6f}',
        formatters={labels['population']: lambda x: f'{x:,}', labels['eigenpair_max_abs_residual']: lambda x: f'{x:.2e}'})
    definitions = dictionary.rename(columns={'column':'CSV column', 'label':'Table heading', 'definition':'Meaning and formula'}).to_html(index=False, border=0)
    parameters_text = ', '.join(f'{key}={value}' for key,value in parameters.items())
    content = '''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NYC segregation comparison — four categories</title>
<style>
body{font:15px/1.55 system-ui,sans-serif;color:#17283b;background:#f4f7fa;margin:0;padding:32px}
main{max-width:1600px;margin:auto}h1{font-size:29px;margin-bottom:8px}h2{font-size:21px}
.box{background:white;border:1px solid #d6dfe8;border-radius:8px;padding:20px;margin:20px 0}
.scroll{overflow:auto;max-height:520px}table{border-collapse:collapse;width:100%}th,td{padding:12px;border-bottom:1px solid #d6dfe8;text-align:left;vertical-align:top}
th{background:#e6eef6}thead th{position:sticky;top:0;z-index:1}.scroll th{min-width:145px}
.scroll td{white-space:nowrap;font-variant-numeric:tabular-nums}.scroll th:first-child,.scroll td:first-child{position:sticky;left:0;background:#e6eef6;z-index:2;min-width:205px}
.scroll thead th:first-child{z-index:3}tbody tr:nth-child(even){background:#f8fafc}a{color:#164e87}
.formula{font-size:18px;font-weight:650}code{overflow-wrap:anywhere}p{max-width:1100px}
@media print{body{padding:0}.scroll{overflow:visible;max-height:none}th{position:static!important}table{font-size:8px}}
</style></head><body><main><h1>NYC segregation: four-category comparison</h1>'''
    content += f'<p>{zone_count} sections · {total_population:,} residents · 2020 Census · local WALK/TRANSIT estimates · no Routes API calls</p>'
    content += '''<div class="box"><div class="formula">Normalized PSSI = (λ − K) / (1 − K)<br>
Adjusted Fryer = S − K<br>Additional rescaled Fryer = (S − K) / (1 − K)</div>
<p>K is the category’s citywide population <strong>fraction</strong>. “Population (%)” is display only.
Hispanic includes any race and overlaps with White, Black and Asian alone. Do not sum these categories as a partition.</p>
<p>The <a href="https://eml.berkeley.edu/~fechenique/published/echenique-fryer-qje.pdf">Echenique–Fryer paper (pp. 24–25)</a>
subtracts a simulated random-assignment baseline, which need not equal K. The S − K column implements the requested population-share adjustment; a paper-style simulated baseline is not computed here.</p></div>
<h2>All comparison columns</h2><p>Scroll horizontally to inspect all columns. Values are rounded for display; the CSV retains full precision.</p>'''
    content += f'<div class="box scroll" tabindex="0" aria-label="Four-category results">{table}</div>'
    content += '<p><a href="four_category_indices.csv">Download results CSV</a> · <a href="four_category_column_dictionary.csv">Download column dictionary</a></p>'
    content += f'<h2>Column definitions</h2><div class="box">{definitions}</div>'
    content += f'<h2>Model interpretation</h2><p>Parameters: {escape(parameters_text)}.</p>'
    content += '''<p>Fryer SSI is the notebook’s majority-section adaptation on the same population-weighted interaction matrix P as PSSI. These 70 sections are not individual people or the original paper’s block-neighbor network. Negative adjusted values are retained. The Asian category has only one majority section in this dataset.</p>
<p>Travel inputs combine approximate walking, scheduled subway/SIR, and labeled transit proxies for missing rail candidates. The analysis uses 2020 demographics and the saved 2026 schedule/fares. Scores depend on these modeling assumptions. Scalar score normalization does not change the right eigenvectors, which separately sum to one.</p>
</main></body></html>'''
    (folder/'four_category_comparison.html').write_text(content)
