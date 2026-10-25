#!/usr/bin/env python3
"""Create an offline catalog, data dictionary, and geographic graph previews."""
from pathlib import Path
import os, json, csv, html
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'data/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import geopandas as gpd
import pandas as pd
from build_graphs import LEVELS, POP, RACES, VAPS, CRS
OUT=ROOT/'outputs/ny_2020'
LABELS={'counties':'Counties','county_subdivisions':'County subdivisions','tracts':'Census tracts','block_groups':'Block groups','blocks':'Census blocks'}

def dictionary():
    rows=[]
    def add(name,typ,unit,desc,source):rows.append(dict(column=name,type=typ,units=unit,description=desc,source=source))
    add('id','integer','index','Zero-based row index in the supplied statewide shapefile; local to each graph.','Source row order')
    add('GEOID20','string','identifier','2020 Census identifier; preserve as text. Lengths: county 5, subdivision 10, tract 11, block group 12, block 15.','TIGER GEOID20')
    add('BLOCKID20','string','identifier','Block-only alias of GEOID20.','TIGER GEOID20')
    add('area','integer','m²','Census land area, excluding water.','TIGER ALAND20')
    add('geometry_area','float','m²','Full projected polygon area, including mapped water.','Geometry in EPSG:5070')
    for c,description in [('C_X','Geometric centroid easting'),('C_Y','Geometric centroid northing')]:add(c,'float','m',description+' in NAD83 / Conus Albers, EPSG:5070.','Projected geometry')
    add('boundary_node','boolean','','Whether the polygon touches the boundary of the union of polygons, including point touches.','GerryChain')
    add('boundary_perim','float','m','Exterior perimeter relative to the union of polygons; zero for interior units.','GerryChain; EPSG:5070')
    add('shared_perim','float','m','Edge attribute: positive length of shared polygon boundary.','GerryChain; EPSG:5070')
    add('UR20','string','category','Blocks only: U = urban, R = rural under the 2020 Census classification.','Updated TIGER2020 block attributes')
    add('UACE20','string','identifier','Blocks only: Census urban-area identifier; blank when unavailable.','Updated TIGER2020 block attributes')
    add('UATYPE20','string','category','Blocks only: urban-area type exactly as supplied by Census.','Updated TIGER2020 block attributes')
    add('INTPTLAT20','string','degrees','Census internal-point latitude, distinct from the geometric centroid.','Original TIGER attributes')
    add('INTPTLON20','string','degrees','Census internal-point longitude, distinct from the geometric centroid.','Original TIGER attributes')
    race_labels=['White alone','Black or African American alone','American Indian or Alaska Native alone','Asian alone','Native Hawaiian or Other Pacific Islander alone','Some other race alone','Two or more races']
    defs={'TOTPOP':('Total population','P0010001 / P0020001'),'HISP':('Hispanic or Latino population, any race','P0020002'),'VAP':('Population age 18 and over','P0030001 / P0040001'),'HVAP':('Hispanic or Latino population age 18 and over','P0040002')}
    for i,(r,label,vap) in enumerate(zip(RACES,race_labels,VAPS)):
        defs['NH_'+r]=(label+', not Hispanic or Latino',f'P002{i+5:04d}')
        defs['H_'+r]=(label+', Hispanic or Latino',f'P001{i+3:04d} - P002{i+5:04d}')
        defs[vap]=(label+', not Hispanic or Latino, age 18 and over',f'P004{i+5:04d}')
    for prefix,qualifier in [('', ''),('U','; counted in urban blocks'),('R','; counted in rural blocks')]:
        for c in POP:
            desc,source=defs[c];add(prefix+c,'integer','persons',desc+qualifier,'2020 PL 94-171: '+source)
    for c,desc in [('component','Connected component number; largest component is 0'),('degree','Number of graph neighbors')]:add(c,'integer','count',desc+'; node-table diagnostic only.','Graph topology')
    for c,desc in [('source','Source node ID'),('target','Target node ID')]:add(c,'integer','index',desc+' in the edge table.','Graph topology')
    for c,desc in [('source_geoid','Source Census GEOID'),('target_geoid','Target Census GEOID')]:add(c,'string','identifier',desc+' in the edge table.','Source attributes')
    with (OUT/'data_dictionary.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    return rows

def draw_maps(report):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titleweight':'bold','savefig.facecolor':'#f7f8f5'})
    for level,kind in LEVELS.items():
        out=OUT/level/'preview.png'
        x=gpd.read_file(ROOT/f'data/raw/tl_2020_36_{kind}20.shp')
        if level in ['tracts','block_groups','blocks']:
            x=x[x.COUNTYFP20.eq('001')]
            extent='Albany County detail'
        else:extent='New York State'
        x=x.to_crs(CRS)
        nodes=pd.read_csv(OUT/level/'nodes.csv',index_col='id',dtype={'GEOID20':str})
        edges=pd.read_csv(OUT/level/'edges.csv')
        nodes=nodes.loc[x.index]
        edges=edges[edges.source.isin(x.index)&edges.target.isin(x.index)]
        fig,ax=plt.subplots(figsize=(8,5.6))
        ax.set_facecolor('#f7f8f5')
        x.geometry.simplify(25 if level=='blocks' else 100).plot(ax=ax,color='#e2e8df',edgecolor='#aebcac',linewidth=.35 if level=='blocks' else .6)
        xy=nodes[['C_X','C_Y']]
        if len(edges):
            segs=list(zip(xy.loc[edges.source].to_numpy(),xy.loc[edges.target].to_numpy()))
            ax.add_collection(LineCollection(segs,colors='#15796e',linewidths=.32 if level=='blocks' else .65,alpha=.75))
        ax.scatter(nodes.C_X,nodes.C_Y,s=1 if level=='blocks' else 5,c='#173d39',zorder=3)
        ax.set_title(LABELS[level],loc='left',fontsize=17,pad=12)
        ax.text(0,1.015,extent,transform=ax.transAxes,ha='left',fontsize=9,color='#5c6a65')
        ax.set_axis_off();fig.tight_layout()
        fig.savefig(out,dpi=150,bbox_inches='tight');plt.close(fig)
        print('Rendered',level,flush=True)

def main():
    report=json.loads((OUT/'validation.json').read_text());rows=dictionary();draw_maps(report)
    cards=[];table=[]
    for level in LEVELS:
        q=report[level]
        links=' · '.join(f'<a href="{level}/{file}" download>{label}</a>' for file,label in [('graph.json','Graph JSON'),('nodes.csv','Nodes CSV'),('edges.csv','Edges CSV')])
        card=f'''<article class="card"><img src="{level}/preview.png" alt="{LABELS[level]} geographic dual graph"><div class="cardbody"><h3>{LABELS[level]}</h3><p>{q['nodes']:,} nodes <span>·</span> {q['edges']:,} edges</p><p class="links">{links}</p><details><summary>Validation and assignments</summary><p>{q['components']} connected component(s); {len(q['isolates'])} isolated node(s).</p><a href="{level}/validation.json">Validation JSON</a> · <a href="{level}/warnings.json">Library warnings</a>'''
        if level!='blocks':card+=f' · <a href="{level}/block_assignment.csv.gz" download>Block assignments</a>'
        cards.append(card+'</details></div></article>')
        table.append(f'<tr><td>{LABELS[level]}</td><td>{q["nodes"]:,}</td><td>{q["edges"]:,}</td><td>{q["components"]}</td><td>{q["TOTPOP"]:,}</td></tr>')
    dictionary_rows=''.join('<tr>'+''.join(f'<td>{html.escape(str(r[c]))}</td>' for c in ['column','units','description','source'])+'</tr>' for r in rows)
    county=json.loads((OUT/'counties/validation.json').read_text())
    page='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>New York · Census Dual Graphs 2020</title><style>
:root{--ink:#16332e;--muted:#5f706a;--line:#dce4dc;--paper:#f7f8f5;--accent:#117769}*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:16px/1.6 system-ui,-apple-system,sans-serif}main{max-width:1160px;margin:auto;padding:52px 28px 70px}nav{display:flex;gap:25px;padding-bottom:24px;border-bottom:1px solid var(--line);font-size:13px;letter-spacing:.05em}a{color:var(--accent);text-underline-offset:3px}h1{font:500 clamp(36px,6vw,62px)/1.06 Georgia,serif;letter-spacing:-.035em;max-width:850px;margin:30px 0 22px}h2{font:500 30px/1.2 Georgia,serif;margin:45px 0 20px}h3{font-size:19px;margin:0}p{margin:12px 0}.lead{max-width:850px;font-size:18px;color:var(--muted)}.eyebrow{margin-top:35px;text-transform:uppercase;letter-spacing:.15em;font-size:12px}.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:1px;background:var(--line);border:1px solid var(--line);margin:30px 0}.stat{background:white;padding:22px}.stat b{display:block;font-size:27px;font-weight:550}.stat span{font-size:13px;color:var(--muted)}.note{padding:17px 20px;background:#e9f0e8;border-left:3px solid var(--accent)}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:22px}.card{background:white;border:1px solid var(--line);border-radius:8px;overflow:hidden}.card img{display:block;width:100%;aspect-ratio:8/5.6;object-fit:contain;background:var(--paper)}.cardbody{padding:20px}.cardbody p{color:var(--muted);margin:6px 0 13px}.cardbody span{padding:0 6px}.links{font-size:14px}details{font-size:13px;border-top:1px solid var(--line);padding-top:10px}summary{cursor:pointer;color:var(--muted)}.scroll{overflow:auto}table{width:100%;border-collapse:collapse;background:white;font-size:14px}th,td{text-align:left;padding:12px 14px;border-bottom:1px solid var(--line);vertical-align:top}th{background:#e9f0e8}code,pre{font-family:ui-monospace,SFMono-Regular,monospace}pre{background:#16332e;color:#eaf2e9;padding:24px;border-radius:6px;overflow:auto;font-size:14px}footer{margin-top:40px;padding-top:20px;border-top:1px solid var(--line);color:var(--muted);font-size:13px}.small{font-size:13px;color:var(--muted)}input{width:100%;padding:12px;margin:12px 0;border:1px solid var(--line);border-radius:4px;font:inherit}#dictionary td:first-child{font-family:monospace;color:var(--accent)}@media(max-width:700px){main{padding:24px 16px}.grid{grid-template-columns:1fr}.stats{grid-template-columns:1fr}.stat{padding:12px 18px}nav{gap:15px}}
</style></head><body><main><nav><strong>APMA 4903 / DATA ATLAS</strong><a href="#downloads">Downloads</a><a href="#methods">Methods</a><a href="#dictionary">Dictionary</a></nav><p class="eyebrow">New York · FIPS 36 · 2020 Census</p><h1>One state.<br>Five geographic scales.</h1><p class="lead">Census dual graphs in the style of Daryl DeFord’s collection. Each geographic unit becomes a node; two nodes are connected when their polygons share a positive-length boundary.</p>'''
    page+=f'''<div class="stats"><div class="stat"><b>{county['TOTPOP']:,}</b><span>Total population · 2020 Census</span></div><div class="stat"><b>{report['blocks']['nodes']:,}</b><span>Blocks matched to population and urban/rural data</span></div><div class="stat"><b>75</b><span>Demographic fields per node · total, urban, rural</span></div></div>'''
    page+='''<p class="note"><strong>2020 adaptation.</strong> DeFord’s published graphs use 2010 geography. These files use 2020 geography and identifiers. The supplied TIGER archive has been supplemented with official Census population counts and updated urban/rural labels.</p><h2 id="downloads">Explore and download</h2><p class="small">Maps show the whole state for counties and subdivisions, and Albany County detail for finer levels. <strong>Every download covers all of New York.</strong> Map lines connect geometric centroids; they illustrate adjacency and are not shared-boundary shapes.</p><div class="grid">'''+''.join(cards)+'''</div><h2>Checks across all five scales</h2><div class="scroll"><table><thead><tr><th>Geography</th><th>Nodes</th><th>Edges</th><th>Components</th><th>Population</th></tr></thead><tbody>'''+''.join(table)+'''</tbody></table></div><p class="small">All 25 base demographic fields match published Census records at every unit. All 75 demographic sums are conserved. Urban + rural = total for every field and unit. All larger-unit assignments match Census identifiers. No artificial connections were added. <a href="validation.json">Validation report</a> · <a href="manifest.json">Input checksums and versions</a></p>'''
    page+=f'''<p>Population in urban blocks: <strong>{county['UTOTPOP']:,}</strong>. Population in rural blocks: <strong>{county['RTOTPOP']:,}</strong>.</p>'''
    page+='''<h2 id="methods">How the graphs were built</h2><p>GerryChain constructs rook adjacency from unsimplified polygons. maup assigns Census blocks to each larger geography; block population is then summed, separately for urban and rural blocks. Source row indices are retained as node IDs, and Census GEOIDs are stored as strings.</p><p>Land area comes directly from <code>ALAND20</code>. Centroids, full polygon areas, and perimeters are measured in <strong>EPSG:5070</strong>: metres for coordinates and lengths, square metres for areas. Census internal points remain separate. Perimeters are projected measurements, not geodesic lengths. Boundary flags refer to the union of each layer, including coastlines and holes.</p><p><code>VAP</code> means population age 18+, not citizen voting-age population. The data use the federal PL 94-171 release, without a separate New York prisoner-reallocation adjustment. Larger units can contain both urban and rural blocks.</p><h2>Open in Python</h2><pre>from gerrychain import Graph

graph = Graph.from_json("tracts/graph.json")</pre><p class="small">Run from this catalog’s directory. JSON uses NetworkX adjacency format. Edges occur in both endpoints’ JSON adjacency lists; the edge CSV lists each undirected edge once. Import CSV GEOIDs as text.</p><h2 id="dictionary">Data dictionary</h2><p>Every demographic field has an unprefixed total, a <code>U</code> urban-block total, and an <code>R</code> rural-block total. Original TIGER attributes are also retained. <a href="data_dictionary.csv" download>Download dictionary CSV</a>.</p><label for="filter" class="small">Filter attributes by name, meaning, or source</label><input id="filter" type="search" placeholder="Try VAP, shared_perim, or urban"><div class="scroll"><table><thead><tr><th>Column</th><th>Units</th><th>Meaning</th><th>Source</th></tr></thead><tbody id="fields">'''+dictionary_rows+'''</tbody></table></div><footer><strong>Sources and reproducibility</strong><p><a href="https://people.csail.mit.edu/ddeford/dual_graphs.html">DeFord’s collection</a> · <a href="https://people.csail.mit.edu/ddeford/data_cols.html">Original schema</a> · <a href="https://github.com/mggg/GerryChain">GerryChain</a> · <a href="https://github.com/mggg/maup">maup</a> · <a href="https://www2.census.gov/programs-surveys/decennial/2020/data/01-Redistricting_File--PL_94-171/New_York/">Census PL 94-171</a> · <a href="https://www2.census.gov/geo/tiger/TIGER2020/TABBLOCK20/">Updated block attributes</a></p><p>Source geometry: the supplied tl_2020_36_all.zip, statewide 2020 layers. Consult the project README and scripts for complete methods. This catalog works offline; only external source links require internet access.</p></footer></main><script>document.getElementById('filter').addEventListener('input',e=>{const q=e.target.value.toLowerCase();document.querySelectorAll('#fields tr').forEach(r=>r.hidden=!r.textContent.toLowerCase().includes(q))});</script></body></html>'''
    (OUT/'index.html').write_text(page)
    print('Catalog written',OUT/'index.html')
if __name__=='__main__':main()
