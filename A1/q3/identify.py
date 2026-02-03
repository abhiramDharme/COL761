import sys
import os
import math
import networkx as nx
import pandas as pd
from collections import Counter
from gspan_mining.gspan import gSpan

# fix for pandas warning
pd.DataFrame.append = lambda self, other, **kw: pd.concat([self, other], **kw)

def parse_dataset(filepath):
    graphs = []
    curr = None
    f = open(filepath, 'r')
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if parts[0] == 't' or line.startswith('#'):
            if curr is not None:
                graphs.append(curr)
            curr = nx.Graph()
        elif parts[0] == 'v':
            if curr is None:
                curr = nx.Graph()
            curr.add_node(int(parts[1]), label=int(parts[2]))
        elif parts[0] == 'e':
            curr.add_edge(int(parts[1]), int(parts[2]), label=int(parts[3]))
    if curr is not None:
        graphs.append(curr)
    f.close()
    return graphs

def run_gspan_with_where(graphs, min_sup, max_vert=8):
    temp_in = f"temp_{os.getpid()}.in"
    f = open(temp_in, 'w')
    for idx, g in enumerate(graphs):
        f.write(f"t # {idx}\n")
        nodes = sorted(g.nodes())
        mapping = {}
        for i, n in enumerate(nodes):
            mapping[n] = i
        for n in nodes:
            f.write(f"v {mapping[n]} {g.nodes[n]['label']}\n")
        for u, v in g.edges():
            f.write(f"e {mapping[u]} {mapping[v]} {g.edges[u,v]['label']}\n")
    f.close()
    
    temp_out = f"temp_{os.getpid()}.out"
    orig_stdout = sys.stdout
    with open(temp_out, 'w') as f:
        sys.stdout = f
        gs = gSpan(temp_in, min_support=min_sup, min_num_vertices=2, max_num_vertices=max_vert, where=True)
        gs.run()
    sys.stdout = orig_stdout
    
    patterns = []
    curr_g = None
    if os.path.exists(temp_out):
        with open(temp_out, 'r') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                if line.startswith('t #'):
                    if curr_g and curr_g.number_of_nodes() > 0: patterns.append(curr_g)
                    curr_g = nx.Graph()
                elif line.startswith('v'): 
                    p = line.split(); curr_g.add_node(int(p[1]), label=int(p[2]))
                elif line.startswith('e'): 
                    p = line.split(); curr_g.add_edge(int(p[1]), int(p[2]), label=int(p[3]))
                elif line.startswith('Support:'):
                    curr_g.graph['support'] = int(line.split(':')[1].strip())
                elif line.startswith('where:'):
                    try: curr_g.graph['where'] = set(eval(line.split(':', 1)[1].strip()))
                    except: curr_g.graph['where'] = set()
            if curr_g and curr_g.number_of_nodes() > 0: patterns.append(curr_g)
    
    for f in [temp_in, temp_out]:
        if os.path.exists(f): os.remove(f)
    return patterns

def pattern_hash(g):
    n_labels = []
    for n in g.nodes():
        n_labels.append(g.nodes[n]['label'])
    n_labels.sort()
    e_labels = []
    for u,v in g.edges():
        e_labels.append((g.nodes[u]['label'], g.nodes[v]['label'], g.edges[u,v]['label']))
    e_labels.sort()
    return (len(g.nodes()), len(g.edges()), tuple(n_labels), tuple(e_labels))

def select_features_best(pool, k=50, total_graphs=0):
    if not pool or total_graphs == 0:
        return pool[:k]
    
    candidates = []
    for g in pool:
        size = g.number_of_edges()
        if size < 2:
            continue
        
        support = g.graph.get('support', 1)
        where = g.graph.get('where', set())
        coverage = support / total_graphs
        
        coverage_score = 1.0 - abs(coverage - 0.25) * 2.0
        if coverage_score < 0.1:
            coverage_score = 0.1
        
        if size == 4:
            size_bonus = math.exp(size * 0.6)
        elif size == 3:
            size_bonus = math.exp(size * 0.45)
        elif size == 2:
            size_bonus = math.exp(size * 0.2)
        else:
            size_bonus = math.exp(size * 0.4)
        
        base_score = coverage_score * size_bonus
        
        candidates.append({
            'graph': g, 'score': base_score, 'where': where,
            'size': size, 'hash': pattern_hash(g)
        })
    
    candidates.sort(key=lambda x: -x['score'])
    print(f"    [{len(candidates)} candidates]")
    
    selected = []
    covered = set()
    seen = set()
    
    for i in range(k):
        if len(candidates) == 0:
            break
        
        best = None
        best_gain = -1
        
        for item in candidates:
            if item['hash'] in seen:
                continue
            
            new_cov = len(item['where'] - covered)
            
            if len(covered) > 0:
                overlap = len(item['where'] & covered) / max(1, len(item['where']))
                anti_corr = 1.0 - overlap * 0.5
            else:
                anti_corr = 1.0
            
            gain = (new_cov + 1) * anti_corr * (item['size'] ** 0.3)
            
            if gain > best_gain:
                best_gain = gain
                best = item
        
        if best is None:
            for item in candidates:
                if item['hash'] not in seen:
                    best = item
                    break
        
        if best:
            selected.append(best['graph'])
            seen.add(best['hash'])
            covered.update(best['where'])
            candidates.remove(best)
    
    return selected

def save_features(features, path):
    with open(path, 'w') as f:
        for idx, g in enumerate(features):
            f.write(f"t # {idx} * 0\n")
            mapping = {n: i for i, n in enumerate(g.nodes())}
            for n in g.nodes():
                f.write(f"v {mapping[n]} {g.nodes[n]['label']}\n")
            for u, v in g.edges():
                f.write(f"e {mapping[u]} {mapping[v]} {g.edges[u,v]['label']}\n")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 identify.py <database> <features>")
        sys.exit(1)
    
    db_path, feat_path = sys.argv[1], sys.argv[2]
    
    print(f"Loading {db_path}...")
    graphs = parse_dataset(db_path)
    N = len(graphs)
    print(f"{N} graphs loaded.")
    
    # 30% support
    support_levels = [0.30, 0.25, 0.20, 0.15]
    max_v = 8
    
    final_pool = []
    for pct in support_levels:
        min_sup = max(50, int(pct * N))
        print(f"Mining at {int(pct*100)}% support (min_sup={min_sup})...")
        patterns = run_gspan_with_where(graphs, min_sup, max_vert=max_v)
        print(f"    -> {len(patterns)} patterns")
        if len(patterns) >= 50:
            final_pool = patterns
            break
    
    if not final_pool:
        final_pool = run_gspan_with_where(graphs, max(1, int(0.10 * N)), max_vert=max_v)
    
    print(f"BEST selection: filter size-1, boost size-4...")
    features = select_features_best(final_pool, 50, N)
    
    sizes = Counter(g.number_of_edges() for g in features)
    print(f"Size distribution: {dict(sorted(sizes.items()))}")
    
    save_features(features, feat_path)
    print(f"Saved to {feat_path}")