#!/usr/bin/env python3
"""
truths_rx.py - Generate ground truth using rustworkx

For each query, find all database graphs that contain the query as a subgraph.
"""

import sys
import pickle
import rustworkx as rx
from common import parse_graph_database

def main():
    if len(sys.argv) != 4:
        print("Usage: python truths_rx.py <db_path> <query_path> <output_truth.dat>")
        sys.exit(1)

    db_path = sys.argv[1]
    q_path = sys.argv[2]
    out_path = sys.argv[3]

    print(f"Loading graphs...")
    db_graphs = parse_graph_database(db_path)
    q_graphs = parse_graph_database(q_path)
    
    print(f"Database: {len(db_graphs)} graphs")
    print(f"Queries: {len(q_graphs)} graphs")
    
    results = {}
    
    for q_idx, query in enumerate(q_graphs):
        if q_idx % 10 == 0:
            print(f"Processing query {q_idx}/{len(q_graphs)}...")
        
        matches = []
        q_nodes = query.num_nodes()
        q_edges = query.num_edges()
        
        for db_idx, db_graph in enumerate(db_graphs):
            # Size pruning
            if db_graph.num_nodes() < q_nodes or db_graph.num_edges() < q_edges:
                continue
            
            # Check subgraph isomorphism
            try:
                is_subgraph = rx.is_subgraph_isomorphic(
                    db_graph, query,
                    node_matcher=lambda x, y: x.get('label') == y.get('label'),
                    edge_matcher=lambda x, y: x.get('label') == y.get('label'),
                    induced=False
                )
                if is_subgraph:
                    matches.append(db_idx)
            except:
                pass
        
        results[q_idx] = matches
        print(f"  Query {q_idx}: {len(matches)} matches")
    
    # Save in the expected format
    print(f"Saving truth to {out_path}...")
    with open(out_path, 'w') as f:
        for q_idx in sorted(results.keys()):
            f.write(f"q # {q_idx}\n")
            f.write("c # " + " ".join(map(str, results[q_idx])) + "\n")
    
    print("Done!")

if __name__ == "__main__":
    main()
