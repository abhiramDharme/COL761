import rustworkx as rx

def parse_graph_database(file_path):
    graphs = []
    current_graph = None
    node_map = {}  
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split()
            tag = parts[0]
            
            if tag.startswith('t') or tag.startswith('#'):
                if current_graph is not None: 
                    graphs.append(current_graph)
                current_graph = rx.PyGraph()
                node_map = {}
            elif tag == 'v':
                if current_graph is None: 
                    current_graph = rx.PyGraph()
                    node_map = {}
                try:
                    node_id = int(parts[1])
                    label = parts[2]
                    rx_idx = current_graph.add_node({'label': label, 'id': node_id})
                    node_map[node_id] = rx_idx
                except: pass
            elif tag == 'e':
                try:
                    src = int(parts[1])
                    dst = int(parts[2])
                    label = parts[3]
                    if src in node_map and dst in node_map:
                        current_graph.add_edge(node_map[src], node_map[dst], {'label': label})
                except: pass

    if current_graph is not None: graphs.append(current_graph)
    return graphs