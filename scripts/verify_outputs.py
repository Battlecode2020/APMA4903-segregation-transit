#!/usr/bin/env python3
"""Independent read-back and sampled spatial checks of saved output files."""
from pathlib import Path
import gc,json
import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
import shapely
from gerrychain import Graph
from build_graphs import ROOT,OUT,LEVELS,POP,ALLPOP,CRS

def main():
    rng=np.random.default_rng(2020); results={}
    for level,kind in LEVELS.items():
        d=OUT/level
        print('Verifying',level,flush=True)
        # Read using GerryChain itself, exercising the intended loading interface.
        graph=Graph.from_json(str(d/'graph.json')).get_nx_graph()
        nodes=pd.read_csv(d/'nodes.csv',index_col='id',dtype={'GEOID20':str,'BLOCKID20':str})
        edges=pd.read_csv(d/'edges.csv',dtype={'source_geoid':str,'target_geoid':str})
        assert len(nodes)==len(graph)
        assert graph.number_of_edges()==len(edges)
        assert not edges.duplicated(['source','target']).any()
        assert (edges.source<edges.target).all()
        assert (edges.shared_perim>0).all()
        assert nodes.GEOID20.is_unique
        assert int(nodes.TOTPOP.sum())==20201249
        assert np.array_equal(nodes[POP].to_numpy(),nodes[['U'+c for c in POP]].to_numpy()+nodes[['R'+c for c in POP]].to_numpy())
        # Check every serialized demographic value and ID against the node table.
        got=pd.DataFrame.from_dict(dict(graph.nodes(data=True)),orient='index')
        assert np.array_equal(got.loc[nodes.index,ALLPOP].to_numpy(),nodes[ALLPOP].to_numpy())
        assert (got.GEOID20==nodes.GEOID20).all()
        assert np.array_equal(nodes.degree.to_numpy(),np.array([graph.degree(i) for i in nodes.index]))
        q=json.loads((d/'validation.json').read_text())
        assert nx.number_connected_components(graph)==q['components']
        assert sorted(nx.isolates(graph))==sorted(q['isolates'])
        source=gpd.read_file(ROOT/f'data/raw/tl_2020_36_{kind}20.shp')
        assert (source.GEOID20.to_numpy()==nodes.GEOID20.to_numpy()).all()
        assert (source.ALAND20.to_numpy()==nodes.area.to_numpy()).all()
        projected=source.to_crs(CRS)
        # Independent native-coordinate rook check: query all candidates for selected nodes.
        # Reprojection is applied to the shared boundary itself for the length check.
        picked=rng.choice(len(source),size=min(250,len(source)),replace=False)
        index=source.sindex; lengths=0
        for i in picked:
            native=source.geometry.iloc[i]
            candidates=index.query(native,predicate='intersects')
            expected=set()
            for j in candidates:
                if i==j:continue
                shared=native.boundary.intersection(source.geometry.iloc[j].boundary)
                if shared.length>0:
                    expected.add(int(j))
                    metric=gpd.GeoSeries([shared],crs=source.crs).to_crs(CRS).length.iloc[0]
                    assert graph.has_edge(int(i),int(j)),f'Missing edge {level}: {i}, {j}'
                    assert np.isclose(metric,graph[int(i)][int(j)]['shared_perim'],rtol=1e-7,atol=1e-5),f'Length mismatch {level}: {i}, {j}'
                    lengths+=1
            assert expected==set(graph.neighbors(int(i))),f'Adjacency mismatch {level}: {i}'
        results[level]={'read_back':'passed','all_serialized_population_values':'passed','source_identifiers_and_land_area':'passed','native_rook_nodes_checked':len(picked),'shared_lengths_checked':lengths}
        del graph,nodes,edges,got,source,projected
        gc.collect()
    (OUT/'readback_validation.json').write_text(json.dumps(results,indent=2)+'\n')
    print('All read-back and sampled spatial checks passed.',flush=True)
if __name__=='__main__':main()
