import sys
import pickle
import networkx as nx
from networkx.algorithms import isomorphism
from multiprocessing import Pool, cpu_count
from functools import partial
from common import parse_graph_database

# Worker function for multiprocessing
def find_matches_for_query(args):
    """
    Checks one query against ALL database graphs.
    """
    q_idx, query, db_graphs = args
    matches = set()
    
    # Pre-configure matcher settings to reuse logic
    # Note: We must instantiate a new matcher for each pair, but we know the config
    nm_kwargs = {
        'node_match': isomorphism.categorical_node_match('label', -1),
        'edge_match': isomorphism.categorical_edge_match('label', -1)
    }

    for db_idx, db_graph in enumerate(db_graphs):
        # Optimization: If query has more nodes/edges than DB, it can't be a subgraph
        if len(query.nodes) > len(db_graph.nodes) or len(query.edges) > len(db_graph.edges):
            continue
            
        nm = isomorphism.GraphMatcher(db_graph, query, **nm_kwargs)
        if nm.subgraph_is_isomorphic():
            matches.add(db_idx)
            
    return (q_idx, matches)

def main():
    if len(sys.argv) != 4:
        print("Usage: python precompute_truth.py <db_path> <query_path> <output_truth.pkl>")
        sys.exit(1)

    db_path = sys.argv[1]
    q_path = sys.argv[2]
    out_path = sys.argv[3]

    print(f"Loading graphs...")
    db_graphs = parse_graph_database(db_path)
    q_graphs = parse_graph_database(q_path)
    
    print(f"Database: {len(db_graphs)} graphs")
    print(f"Queries: {len(q_graphs)} graphs")
    print(f"Starting brute-force check on {cpu_count()} cores...")

    # Prepare arguments for parallel processing
    # We pass the full db_graphs list to each worker. 
    # On Linux/Mac (fork), this is efficient. On Windows, it adds pickling overhead but is still faster.
    tasks = [(i, q, db_graphs) for i, q in enumerate(q_graphs)]

    true_results = {}
    
    # Use Multiprocessing Pool
    with Pool(processes=cpu_count()) as pool:
        # imap_unordered is often faster as it yields results as soon as they are ready
        for i, (q_idx, matches) in enumerate(pool.imap_unordered(find_matches_for_query, tasks)):
            true_results[q_idx] = matches
            if (i+1) % 10 == 0:
                print(f"Processed {i+1}/{len(q_graphs)} queries...")

    print(f"Finished. Saving ground truth to {out_path}...")
    with open(out_path, 'wb') as f:
        pickle.dump(true_results, f)
    print("Done!")

if __name__ == "__main__":
    main()