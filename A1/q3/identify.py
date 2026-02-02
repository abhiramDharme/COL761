import sys
import pickle
from collections import Counter
import rustworkx as rx

from common import parse_graph_database


# -------------------------
# gIndex-safe feature mining
# -------------------------

def mine_node_label_features(graphs):
    freq = Counter()
    for g in graphs:
        seen = set()
        for data in g.nodes():
            seen.add(data['label'])
        for lbl in seen:
            freq[('node', lbl)] += 1
    return freq


def mine_edge_label_features(graphs):
    freq = Counter()
    for g in graphs:
        seen = set()
        nodes = g.nodes()
        for u, v in g.edge_list():
            lu = nodes[u]['label']
            lv = nodes[v]['label']
            el = g.get_edge_data(u, v)['label']
            if lu <= lv:
                key = ('edge', lu, el, lv)
            else:
                key = ('edge', lv, el, lu)
            seen.add(key)
        for k in seen:
            freq[k] += 1
    return freq


def mine_path3_features(graphs):
    freq = Counter()
    for g in graphs:
        seen = set()
        nodes = g.nodes()
        adj = {i: [] for i in range(len(nodes))}
        for u, v in g.edge_list():
            adj[u].append(v)
            adj[v].append(u)

        for v in range(len(nodes)):
            lv = nodes[v]['label']
            neigh = adj[v]
            for i in range(len(neigh)):
                for j in range(i + 1, len(neigh)):
                    u = neigh[i]
                    w = neigh[j]
                    lu = nodes[u]['label']
                    lw = nodes[w]['label']
                    e1 = g.get_edge_data(v, u)['label']
                    e2 = g.get_edge_data(v, w)['label']
                    if (lu, e1) <= (lw, e2):
                        key = ('path3', lu, e1, lv, e2, lw)
                    else:
                        key = ('path3', lw, e2, lv, e1, lu)
                    seen.add(key)

        for k in seen:
            freq[k] += 1
    return freq

def mine_path4_features(graphs):
    freq = Counter()

    for g in graphs:
        seen = set()
        nodes = g.nodes()

        # adjacency list
        adj = {i: [] for i in range(len(nodes))}
        for u, v in g.edge_list():
            adj[u].append(v)
            adj[v].append(u)

        for b in range(len(nodes)):
            for c in adj[b]:
                if c == b:
                    continue
                for a in adj[b]:
                    if a == c:
                        continue
                    for d in adj[c]:
                        if d == b or d == a:
                            continue

                        la = nodes[a]['label']
                        lb = nodes[b]['label']
                        lc = nodes[c]['label']
                        ld = nodes[d]['label']

                        e_ab = g.get_edge_data(a, b)['label']
                        e_bc = g.get_edge_data(b, c)['label']
                        e_cd = g.get_edge_data(c, d)['label']

                        forward = ('path4', la, e_ab, lb, e_bc, lc, e_cd, ld)
                        backward = ('path4', ld, e_cd, lc, e_bc, lb, e_ab, la)

                        seen.add(min(forward, backward))

        for k in seen:
            freq[k] += 1

    return freq

def mine_cycles(graphs, cycle_len):
    freq = Counter()

    for g in graphs:
        seen = set()
        nodes = g.nodes()

        def dfs(start, current, visited, path_edges):
            if len(visited) == cycle_len:
                if g.has_edge(current, start):
                    labels = sorted(nodes[n]['label'] for n in visited)

                    edge_labels = [
                        g.get_edge_data(u, v)['label']
                        for (u, v) in path_edges
                    ]

                    edge_labels.append(
                        g.get_edge_data(current, start)['label']
                    )

                    edge_labels = sorted(edge_labels)

                    key = (f'cycle{cycle_len}', tuple(labels), tuple(edge_labels))
                    seen.add(key)
                return

            for nbr in g.neighbors(current):
                if nbr not in visited and len(visited) < cycle_len:
                    edge_label = g.get_edge_data(current, nbr)['label']
                    dfs(
                        start,
                        nbr,
                        visited + [nbr],
                        path_edges + [(current, nbr)]
                    )

        for v in range(len(nodes)):
            dfs(v, v, [v], [])

        for k in seen:
            freq[k] += 1

    return freq

# -------------------------
# Feature selection by support window
# -------------------------

def select_by_variance(freq_map, N, k_max):
    scored = []
    for f, c in freq_map.items():
        p = c / N
        score = p * (1 - p)
        if score > 0:
            scored.append((score, f))

    scored.sort(reverse=True)
    return [f for _, f in scored[:k_max]]


# -------------------------
# Build rustworkx feature graphs
# -------------------------

def build_feature_graphs(features):
    out = []
    for feat in features:
        if feat[0] == 'node':
            _, lbl = feat
            g = rx.PyGraph()
            g.add_node({'label': lbl})
            out.append(g)

        elif feat[0] == 'edge':
            _, l1, el, l2 = feat
            g = rx.PyGraph()
            n0 = g.add_node({'label': l1})
            n1 = g.add_node({'label': l2})
            g.add_edge(n0, n1, {'label': el})
            out.append(g)

        elif feat[0] == 'path3':
            _, l1, e1, l2, e2, l3 = feat
            g = rx.PyGraph()
            n0 = g.add_node({'label': l1})
            n1 = g.add_node({'label': l2})
            n2 = g.add_node({'label': l3})
            g.add_edge(n0, n1, {'label': e1})
            g.add_edge(n1, n2, {'label': e2})
            out.append(g)

        elif feat[0] == 'path4':
            _, l1, e1, l2, e2, l3, e3, l4 = feat
            g = rx.PyGraph()
            n0 = g.add_node({'label': l1})
            n1 = g.add_node({'label': l2})
            n2 = g.add_node({'label': l3})
            n3 = g.add_node({'label': l4})
            g.add_edge(n0, n1, {'label': e1})
            g.add_edge(n1, n2, {'label': e2})
            g.add_edge(n2, n3, {'label': e3})
            out.append(g)

        elif feat[0].startswith('cycle'):
            _, node_labels, edge_labels = feat
            g = rx.PyGraph()

            nodes = [g.add_node({'label': l}) for l in node_labels]
            for i in range(len(nodes)):
                g.add_edge(
                    nodes[i],
                    nodes[(i+1) % len(nodes)],
                    {'label': edge_labels[i]}
                )
            out.append(g)
    return out


# -------------------------
# Main
# -------------------------

def main():
    if len(sys.argv) != 3:
        print("Usage: python identify.py <graph_dataset> <output_features>")
        sys.exit(1)

    graph_path = sys.argv[1]
    out_path = sys.argv[2]

    graphs = parse_graph_database(graph_path)
    N = len(graphs)

    node_freq = mine_node_label_features(graphs)
    edge_freq = mine_edge_label_features(graphs)
    path3_freq = mine_path3_features(graphs)
    path4_freq = mine_path4_features(graphs)
    cycle5_freq = mine_cycles(graphs, 5)
    cycle6_freq = mine_cycles(graphs, 6)

    node_feats = [f for f, _ in node_freq.most_common(20)]
    edge_feats = [f for f, _ in edge_freq.most_common(30)]

    path3_feats = select_by_variance(path3_freq, N, k_max=90)
    path4_feats = select_by_variance(path4_freq, N, k_max=160)

    cycle5_feats = select_by_variance(cycle5_freq, N, k_max=40)
    cycle6_feats = select_by_variance(cycle6_freq, N, k_max=50)

    selected = (
        node_feats +
        edge_feats +
        path3_feats +
        path4_feats +
        cycle5_feats +
        cycle6_feats
    )
    feature_graphs = build_feature_graphs(selected)

    with open(out_path, "wb") as f:
        pickle.dump(feature_graphs, f)


if __name__ == "__main__":
    main()