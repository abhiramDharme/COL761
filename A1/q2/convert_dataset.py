#!/usr/bin/env python3
"""
Convert Yeast dataset to formats required by gSpan, FSG, and Gaston.
All labels are mapped to integers for maximum performance and compatibility.
"""

import sys
import os

# Fixed label mapping from Piazza instructions
LABEL_ORDER = ['Br', 'C', 'Cl', 'F', 'H', 'I', 'N', 'O', 'P', 'S', 'Si']
LABEL_MAP = {label: idx for idx, label in enumerate(LABEL_ORDER)}

# Edge labels are still auto-generated
edge_label_map = {}
next_edge_label_id = 0

def get_node_label_id(label: str) -> int:
    """Use fixed label mapping from Piazza."""
    if label not in LABEL_MAP:
        raise ValueError(f"Unknown node label: {label}. Expected one of {LABEL_ORDER}")
    return LABEL_MAP[label]

def get_edge_label_id(label: str) -> int:
    """Auto-generate edge label IDs."""
    global next_edge_label_id
    if label not in edge_label_map:
        edge_label_map[label] = next_edge_label_id
        next_edge_label_id += 1
    return edge_label_map[label]

def parse_yeast_dataset(filepath: str) -> list:
    graphs = []
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line.startswith('#'):
            graph_id = line[1:].strip()
            i += 1
            num_nodes = int(lines[i].strip())
            i += 1
            nodes = []
            for _ in range(num_nodes):
                nodes.append(lines[i].strip())
                i += 1
            num_edges = int(lines[i].strip())
            i += 1
            edges = []
            for _ in range(num_edges):
                parts = lines[i].strip().replace(',', ' ').split()
                src, dst = int(parts[0]), int(parts[1])
                etype = parts[2] if len(parts) > 2 else '0'
                edges.append((src, dst, etype))
                i += 1
            graphs.append({'id': graph_id, 'nodes': nodes, 'edges': edges})
        else:
            i += 1
    return graphs

def convert_to_gspan(graphs: list, output_path: str):
    """gSpan integer format"""
    with open(output_path, 'w') as f:
        for idx, graph in enumerate(graphs):
            f.write(f"t # {idx}\n")
            for node_id, label in enumerate(graph['nodes']):
                f.write(f"v {node_id} {get_node_label_id(label)}\n")
            for src, dst, etype in graph['edges']:
                f.write(f"e {src} {dst} {get_edge_label_id(str(etype))}\n")

def convert_to_fsg(graphs: list, output_path: str):
    """FSG integer format"""
    with open(output_path, 'w') as f:
        for idx, graph in enumerate(graphs):
            f.write(f"t # {idx}\n")
            for node_id, label in enumerate(graph['nodes']):
                f.write(f"v {node_id} {get_node_label_id(label)}\n")
            # FSG undirected 'u'
            for src, dst, etype in graph['edges']:
                f.write(f"u {src} {dst} {get_edge_label_id(str(etype))}\n")

def convert_to_gaston(graphs: list, output_path: str):
    """Gaston integer format"""
    with open(output_path, 'w') as f:
        for idx, graph in enumerate(graphs):
            f.write(f"t # {idx}\n")
            for node_id, label in enumerate(graph['nodes']):
                f.write(f"v {node_id} {get_node_label_id(label)}\n")
            for src, dst, etype in graph['edges']:
                f.write(f"e {src} {dst} {get_edge_label_id(str(etype))}\n")

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 convert_dataset.py <input> <out_dir>")
        sys.exit(1)
    
    graphs = parse_yeast_dataset(sys.argv[1])
    out_dir = sys.argv[2]
    
    convert_to_gspan(graphs, os.path.join(out_dir, "yeast_gspan.txt"))
    convert_to_fsg(graphs, os.path.join(out_dir, "yeast_fsg.txt"))
    convert_to_gaston(graphs, os.path.join(out_dir, "yeast_gaston.txt"))
    print(f"Conversion complete. Used fixed label mapping: {len(LABEL_MAP)} node labels, {next_edge_label_id} edge labels.")

if __name__ == "__main__":
    main()
