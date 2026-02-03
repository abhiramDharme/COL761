import sys
import numpy as np
import networkx as nx
from networkx.algorithms import isomorphism
from collections import Counter
from utils import parse_dataset, parse_gspan_output
from joblib import Parallel, delayed

def check_subgraph_monomorphism(G, sub):
    if G.number_of_nodes() < sub.number_of_nodes():
        return False
    if G.number_of_edges() < sub.number_of_edges():
        return False
    
    # label count check
    g_labels = Counter([G.nodes[n]['label'] for n in G.nodes()])
    s_labels = Counter([sub.nodes[n]['label'] for n in sub.nodes()])
    for label, count in s_labels.items():
        if g_labels.get(label, 0) < count:
            return False
    
    g_edges = Counter([G.edges[u,v]['label'] for u, v in G.edges()])
    s_edges = Counter([sub.edges[u,v]['label'] for u, v in sub.edges()])
    for label, count in s_edges.items():
        if g_edges.get(label, 0) < count:
            return False
    
    # actual isomorphism check
    nm = isomorphism.categorical_node_match('label', -1)
    em = isomorphism.categorical_edge_match('label', -1)
    gm = isomorphism.GraphMatcher(G, sub, node_match=nm, edge_match=em)
    return gm.subgraph_is_monomorphic()

def process_single_graph(g_idx, g, features):
    row = np.zeros(len(features), dtype=int)
    for j, feat in enumerate(features):
        if check_subgraph_monomorphism(g, feat):
            row[j] = 1
    return row

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: convert.py <dataset> <features> <output.npy>")
        sys.exit(1)

    graphs = parse_dataset(sys.argv[1])
    features = parse_gspan_output(sys.argv[2])
    output_path = sys.argv[3]
    
    N = len(graphs)
    K = len(features)
    
    print(f"Indexing {N} graphs x {K} features...")
    
    rows = Parallel(n_jobs=-1, backend="loky")(
        delayed(process_single_graph)(i, g, features) for i, g in enumerate(graphs)
    )
    
    matrix = np.array(rows, dtype=int)
    np.save(output_path, matrix)
    print(f"Saved {matrix.shape}")