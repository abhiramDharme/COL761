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
    """
    Select k discriminative subgraph features from mined patterns.
    
    Key improvements:
    1. Filter out high-density features (>70% density) - provide weak filtering
    2. Penalize very rare features (<10% density) - not enough coverage
    3. Prefer features near 25% density for optimal filtering
    4. Maximize diversity: avoid features that filter same graphs
    """
    if not pool or total_graphs == 0:
        return pool[:k]
    
    candidates = []
    filtered_high = 0
    filtered_low = 0
    
    for g in pool:
        size = g.number_of_edges()
        if size < 2:
            continue
        
        support = g.graph.get('support', 1)
        where = g.graph.get('where', set())
        coverage = support / total_graphs
        
        # IMPROVEMENT 1: Filter out high-density features (>55%)
        # Features present in majority of graphs provide weak filtering
        if coverage > 0.55:
            filtered_high += 1
            continue
        
        # IMPROVEMENT 2: Filter very rare features (<10%)
        # These don't appear in enough graphs to be useful
        if coverage < 0.10:
            filtered_low += 1
            continue
        
        # Optimal coverage is around 25% - good balance of filtering power
        # Score peaks at 25% and decreases towards 10% and 80%
        coverage_score = 1.0 - abs(coverage - 0.25) * 2.0
        if coverage_score < 0.1:
            coverage_score = 0.1
        
        # Bonus for larger patterns (more specific, better filtering)
        # Medium patterns (4-6 edges) often provide best balance
        if size >= 6:
            size_bonus = math.exp(size * 0.55)
        elif size == 5:
            size_bonus = math.exp(size * 0.65)
        elif size == 4:
            size_bonus = math.exp(size * 0.7)
        elif size == 3:
            size_bonus = math.exp(size * 0.5)
        else:  # size == 2
            size_bonus = math.exp(size * 0.25)
        
        base_score = coverage_score * size_bonus
        
        candidates.append({
            'graph': g, 'score': base_score, 'where': where,
            'size': size, 'hash': pattern_hash(g), 'coverage': coverage
        })
    
    print(f"    [{len(candidates)} valid candidates, filtered {filtered_high} high-density, {filtered_low} low-density]")
    
    # Sort by base score for initial ordering
    candidates.sort(key=lambda x: -x['score'])
    
    selected = []
    covered = set()  # Graphs already covered by selected features
    seen = set()     # Pattern hashes to avoid duplicates
    
    # IMPROVEMENT 3: Greedy selection maximizing diversity
    for i in range(k):
        if len(candidates) == 0:
            break
        
        best = None
        best_gain = -1
        
        for item in candidates:
            if item['hash'] in seen:
                continue
            
            # Count how many NEW graphs this feature covers
            new_cov = len(item['where'] - covered)
            
            # IMPROVEMENT 4: Moderate anti-correlation scoring
            # Penalize features that overlap with already-covered graphs but not too strongly
            if len(covered) > 0 and len(item['where']) > 0:
                overlap_ratio = len(item['where'] & covered) / len(item['where'])
                # Anti-correlation: moderate penalty for overlap
                anti_corr = 1.0 - overlap_ratio * 0.6
            else:
                anti_corr = 1.0
            
            # Prefer wide range of coverage for better filtering
            coverage_bonus = 1.0
            if 0.10 <= item['coverage'] <= 0.50:
                coverage_bonus = 1.2  # Good range for filtering
            
            # Combined gain score
            gain = (new_cov + 1) * anti_corr * (item['size'] ** 0.5) * coverage_bonus
            
            if gain > best_gain:
                best_gain = gain
                best = item
        
        # Fallback: just pick any unseen pattern
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
    
    # Mine at decreasing support levels until we get enough patterns
    support_levels = [0.30, 0.25, 0.20, 0.15, 0.10]
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
        min_sup = max(1, int(0.05 * N))
        print(f"Mining at 5% support (min_sup={min_sup})...")
        final_pool = run_gspan_with_where(graphs, min_sup, max_vert=max_v)
    
    print(f"Selecting 50 best features...")
    features = select_features_best(final_pool, 50, N)
    
    sizes = Counter(g.number_of_edges() for g in features)
    print(f"Size distribution: {dict(sorted(sizes.items()))}")
    
    save_features(features, feat_path)
    print(f"Saved to {feat_path}")