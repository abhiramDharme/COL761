#!/usr/bin/env python3
"""
generate_queries.py - Generate query graphs by extracting subgraphs from database

For each query, randomly select a database graph and extract a connected subgraph.
"""

import sys
import random
import rustworkx as rx
from common import parse_graph_database

def extract_random_subgraph(graph, min_nodes=5, max_nodes=15):
    """Extract a random connected subgraph from a graph."""
    n = graph.num_nodes()
    if n < min_nodes:
        return graph  # Return entire graph if too small
    
    # Target size
    target_size = random.randint(min_nodes, min(max_nodes, n))
    
    # Start from random node and grow via BFS
    node_indices = list(graph.node_indices())
    start = random.choice(node_indices)
    
    selected = {start}
    frontier = list(graph.neighbors(start))
    random.shuffle(frontier)
    
    while len(selected) < target_size and frontier:
        next_node = frontier.pop(0)
        if next_node not in selected:
            selected.add(next_node)
            new_neighbors = [n for n in graph.neighbors(next_node) if n not in selected]
            random.shuffle(new_neighbors)
            frontier.extend(new_neighbors)
    
    # Create subgraph
    selected_list = list(selected)
    subgraph = rx.PyGraph()
    node_map = {}
    
    for old_idx in selected_list:
        new_idx = subgraph.add_node(graph[old_idx])
        node_map[old_idx] = new_idx
    
    for edge in graph.edge_list():
        if edge[0] in selected and edge[1] in selected:
            edge_data = graph.get_edge_data(edge[0], edge[1])
            subgraph.add_edge(node_map[edge[0]], node_map[edge[1]], edge_data)
    
    return subgraph

def save_graphs(graphs, output_path):
    """Save graphs in standard format."""
    with open(output_path, 'w') as f:
        for idx, g in enumerate(graphs):
            f.write(f"t # {idx}\n")
            for node_idx in g.node_indices():
                label = g[node_idx].get('label', '0')
                f.write(f"v {node_idx} {label}\n")
            
            seen_edges = set()
            for edge in g.edge_list():
                edge_key = (min(edge[0], edge[1]), max(edge[0], edge[1]))
                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    edge_data = g.get_edge_data(edge[0], edge[1])
                    label = edge_data.get('label', '0') if edge_data else '0'
                    f.write(f"e {edge[0]} {edge[1]} {label}\n")

def main():
    if len(sys.argv) < 4:
        print("Usage: python3 generate_queries.py <db_file> <num_queries> <output_file>")
        sys.exit(1)
    
    db_path = sys.argv[1]
    num_queries = int(sys.argv[2])
    output_path = sys.argv[3]
    
    print(f"Loading database from {db_path}...")
    graphs = parse_graph_database(db_path)
    print(f"Loaded {len(graphs)} graphs")
    
    print(f"Generating {num_queries} query subgraphs...")
    queries = []
    sources = []  # Track which DB graph each query came from
    
    random.seed(42)  # For reproducibility
    
    for i in range(num_queries):
        # Pick a random database graph
        db_idx = random.randint(0, len(graphs) - 1)
        db_graph = graphs[db_idx]
        
        # Extract random subgraph
        query = extract_random_subgraph(db_graph, min_nodes=5, max_nodes=12)
        queries.append(query)
        sources.append(db_idx)
        
        if (i + 1) % 10 == 0:
            print(f"  Generated {i + 1}/{num_queries} queries")
    
    # Save queries
    save_graphs(queries, output_path)
    print(f"Saved {len(queries)} queries to {output_path}")
    
    # Save source mapping for debugging
    sources_path = output_path.replace('.txt', '_sources.txt')
    with open(sources_path, 'w') as f:
        for i, src in enumerate(sources):
            f.write(f"{i}: {src}\n")
    print(f"Saved source mapping to {sources_path}")

if __name__ == "__main__":
    main()
