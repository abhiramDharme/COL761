import sys
import pickle
import numpy as np
import rustworkx as rx
from common import parse_graph_database
import collections

def get_node_label_counts(graph):
    counts = collections.Counter()
    for node_idx in graph.node_indices():
        label = graph[node_idx].get('label', '')
        counts[label] += 1
    return counts

def get_edge_label_counts(graph):
    counts = collections.Counter()
    for edge in graph.edge_list():
        edge_data = graph.get_edge_data(edge[0], edge[1])
        label = edge_data.get('label', '')
        counts[label] += 1
    return counts

def get_degree_distribution(graph):
    degrees = [graph.degree(n) for n in graph.node_indices()]
    return collections.Counter(degrees)

def get_2hop_patterns(graph):
    patterns = set()
    
    for node in graph.node_indices():
        node_label = graph[node].get('label', '')
        neighbors = graph.neighbors(node)
        
        for neighbor in neighbors:
            edge_data = graph.get_edge_data(node, neighbor)
            edge_label = edge_data.get('label', '') if edge_data else ''
            neighbor_label = graph[neighbor].get('label', '')
            pattern = f"{node_label}-{edge_label}-{neighbor_label}"
            patterns.add(pattern)
            
            # 2-hop
            neighbors2 = graph.neighbors(neighbor)
            for neighbor2 in neighbors2:
                if neighbor2 != node:
                    edge_data2 = graph.get_edge_data(neighbor, neighbor2)
                    edge_label2 = edge_data2.get('label', '') if edge_data2 else ''
                    neighbor2_label = graph[neighbor2].get('label', '')
                    parts = [node_label, edge_label, neighbor_label, edge_label2, neighbor2_label]
                    pattern = '-'.join(parts)
                    patterns.add(pattern)
    
    return patterns

def get_triangle_patterns(graph):
    triangles = set()
    
    for node in graph.node_indices():
        neighbors = list(graph.neighbors(node))
        
        for i, n1 in enumerate(neighbors):
            for n2 in neighbors[i+1:]:
                if graph.has_edge(n1, n2):
                    labels = sorted([
                        graph[node].get('label', ''),
                        graph[n1].get('label', ''),
                        graph[n2].get('label', '')
                    ])
                    
                    edge_labels = []
                    for e1, e2 in [(node, n1), (n1, n2), (n2, node)]:
                        edge_data = graph.get_edge_data(e1, e2)
                        edge_labels.append(edge_data.get('label', '') if edge_data else '')
                    edge_labels.sort()
                    
                    pattern = f"TRI:{','.join(labels)}|{','.join(edge_labels)}"
                    triangles.add(pattern)
    
    return triangles

def get_star_patterns(graph):
    stars = set()
    
    for node in graph.node_indices():
        neighbors = list(graph.neighbors(node))
        if len(neighbors) >= 3:
            node_label = graph[node].get('label', '')
            neighbor_labels = sorted([graph[n].get('label', '') for n in neighbors])
            edge_labels = sorted([graph.get_edge_data(node, n).get('label', '') 
                                 if graph.get_edge_data(node, n) else '' for n in neighbors])
            
            for k in [3, 4, 5]:
                if len(neighbors) >= k:
                    pattern = f"STAR{k}:{node_label}|{','.join(neighbor_labels[:k])}|{','.join(edge_labels[:k])}"
                    stars.add(pattern)
    
    return stars

def get_4node_patterns(graph):
    patterns = set()
    
    for node in graph.node_indices():
        neighbors1 = list(graph.neighbors(node))
        
        for n1 in neighbors1:
            neighbors2 = [n for n in graph.neighbors(n1) if n != node]
            
            for n2 in neighbors2:
                neighbors3 = [n for n in graph.neighbors(n2) if n != n1]
                
                for n3 in neighbors3:
                    labels = [
                        graph[node].get('label', ''),
                        graph[n1].get('label', ''),
                        graph[n2].get('label', ''),
                        graph[n3].get('label', '')
                    ]
                    
                    forward = '-'.join(labels)
                    backward = '-'.join(reversed(labels))
                    pattern = min(forward, backward)
                    patterns.add(f"PATH4:{pattern}")
                    
                    if graph.has_edge(n3, node):
                        sorted_labels = tuple(sorted(labels))
                        patterns.add(f"CYCLE4:{','.join(sorted_labels)}")
    
    return patterns

def extract_features_for_graph(graph):
    features = {}
    
    features['num_nodes'] = graph.num_nodes()
    features['num_edges'] = graph.num_edges()
    
    n = graph.num_nodes()
    m = graph.num_edges()
    max_edges = n * (n - 1) / 2 if n > 1 else 1
    features['density'] = m / max_edges if max_edges > 0 else 0
    
    node_labels = get_node_label_counts(graph)
    for label, count in node_labels.items():
        features[f'node_label_{label}'] = count
    
    edge_labels = get_edge_label_counts(graph)
    for label, count in edge_labels.items():
        features[f'edge_label_{label}'] = count
    
    degrees = get_degree_distribution(graph)
    for degree, count in degrees.items():
        features[f'degree_{degree}'] = count
    
    for pattern in get_2hop_patterns(graph):
        features[f'2hop_{pattern}'] = 1
    
    for pattern in get_triangle_patterns(graph):
        features[pattern] = 1
    
    for pattern in get_star_patterns(graph):
        features[pattern] = 1
    
    for pattern in get_4node_patterns(graph):
        features[pattern] = 1
    
    return features

def convert_to_feature_matrix(graphs, selected_features):
    print(f"Converting {len(graphs)} graphs using subgraph isomorphism...")

    feature_matrix = np.zeros((len(graphs), len(selected_features)), dtype=np.int8)

    for i, g in enumerate(graphs):
        if i % 50 == 0:
            print(f"Processing graph {i}/{len(graphs)}")

        for j, f in enumerate(selected_features):
            is_present = rx.is_subgraph_isomorphic(
                g,
                f,
                node_matcher=lambda x, y: x['label'] == y['label'],
                edge_matcher=lambda x, y: x['label'] == y['label'],
                induced=False,
            )
            feature_matrix[i, j] = 1 if is_present else 0

    return feature_matrix

def main():
    if len(sys.argv) != 4:
        print("Usage: python convert.py <graphs.txt> <features.pkl> <output.npy>")
        sys.exit(1)
    
    graphs_path = sys.argv[1]
    features_path = sys.argv[2]
    output_path = sys.argv[3]
    
    print("Loading features...")
    with open(features_path, 'rb') as f:
        selected_features = pickle.load(f)
    print(f"Loaded {len(selected_features)} features")
    
    print("Parsing graphs...")
    graphs = parse_graph_database(graphs_path)
    print(f"Loaded {len(graphs)} graphs")
    
    feature_matrix = convert_to_feature_matrix(graphs, selected_features)
    
    print(f"\nFeature matrix shape: {feature_matrix.shape}")
    print(f"Non-zero entries: {np.count_nonzero(feature_matrix)}")
    print(f"Average features per graph: {feature_matrix.sum(axis=1).mean():.2f}")
    
    # Save
    np.save(output_path, feature_matrix)
    print(f"Saved feature matrix to {output_path}")

if __name__ == "__main__":
    main()