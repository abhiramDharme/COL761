import networkx as nx

def parse_dataset(filepath):
    graphs = []
    curr = None
    gid = 0
    f = open(filepath, 'r')
    for line in f:
        line = line.strip()
        if not line:
            continue
        if line.startswith('#'):
            if curr is not None:
                graphs.append(curr)
            curr = nx.Graph(id=gid)
            gid += 1
        elif line.startswith('v'):
            parts = line.split()
            curr.add_node(int(parts[1]), label=int(parts[2]))
        elif line.startswith('e'):
            parts = line.split()
            u = int(parts[1])
            v = int(parts[2])
            label = int(parts[3])
            if not curr.has_edge(u, v):
                curr.add_edge(u, v, label=label)
    if curr is not None:
        graphs.append(curr)
    f.close()
    return graphs

def parse_gspan_output(filepath):
    patterns = []
    curr = None
    f = open(filepath, 'r')
    for line in f:
        line = line.strip()
        if not line:
            continue
        if line.startswith('t #'):
            if curr is not None:
                patterns.append(curr)
            curr = nx.Graph()
        elif line.startswith('v'):
            parts = line.split()
            curr.add_node(int(parts[1]), label=int(parts[2]))
        elif line.startswith('e'):
            parts = line.split()
            curr.add_edge(int(parts[1]), int(parts[2]), label=int(parts[3]))
    if curr is not None:
        patterns.append(curr)
    f.close()
    return patterns