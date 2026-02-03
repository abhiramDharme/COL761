#!/usr/bin/env python3
"""Verify feature extraction consistency."""
import pickle
import rustworkx as rx
from common import parse_graph_database

# Load features
with open('features_muta.pkl', 'rb') as f:
    features = pickle.load(f)

# Load graphs
graphs = parse_graph_database('q3_datasets/Mutagenicity/graphs.txt')

print(f"Loaded {len(features)} features and {len(graphs)} graphs")

# Check feature 0 (which is supposedly inconsistent between query 0 and db[4])
# According to the debug: q0[0]=1 but db4[0]=0 for feature index 0

feature_pattern = features[0]
query_graph = graphs[0]  # Assuming query 0 = graph 0
db_graph_4 = graphs[4]

print(f"\nFeature 0: {feature_pattern.num_nodes()} nodes, {feature_pattern.num_edges()} edges")

# Check if feature 0 is in query graph
in_query = rx.is_subgraph_isomorphic(
    query_graph, feature_pattern,
    node_matcher=lambda x, y: x['label'] == y['label'],
    edge_matcher=lambda x, y: x['label'] == y['label'],
    induced=False
)
print(f"Feature 0 in query (graph 0): {in_query}")

# Check if feature 0 is in db graph 4
in_db4 = rx.is_subgraph_isomorphic(
    db_graph_4, feature_pattern,
    node_matcher=lambda x, y: x['label'] == y['label'],
    edge_matcher=lambda x, y: x['label'] == y['label'],
    induced=False
)
print(f"Feature 0 in db[4]: {in_db4}")

# Now check if query is a subgraph of db[4] (should be True per ground truth)
query_in_db4 = rx.is_subgraph_isomorphic(
    db_graph_4, query_graph,
    node_matcher=lambda x, y: x['label'] == y['label'],
    edge_matcher=lambda x, y: x['label'] == y['label'],
    induced=False
)
print(f"\nQuery (graph 0) is subgraph of db[4]: {query_in_db4}")

# Sizes
print(f"\nQuery: {query_graph.num_nodes()} nodes, {query_graph.num_edges()} edges")
print(f"DB[4]: {db_graph_4.num_nodes()} nodes, {db_graph_4.num_edges()} edges")
