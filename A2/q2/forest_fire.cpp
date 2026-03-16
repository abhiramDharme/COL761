/*
 * COL761 Assignment 2 - Q2: Forest Fire Route Blocking
 * Deterministic SAA + CELF solver with hop-aware reachability.
 *
 * Compile: g++ -O3 -std=c++17 -march=native -flto -o forest_fire forest_fire.cpp
 * Usage:   ./forest_fire <graph> <seeds> <output> <k> <n_sims> <hops>
 */

#ifdef __has_include
  #if __has_include(<bits/stdc++.h>)
    #include <bits/stdc++.h>
  #else
    #include <algorithm>
    #include <chrono>
    #include <cmath>
    #include <cstdint>
    #include <cstdlib>
    #include <fstream>
    #include <iostream>
    #include <numeric>
    #include <queue>
        #include <random>
    #include <set>
    #include <sstream>
    #include <string>
    #include <unordered_set>
    #include <vector>
  #endif
#else
  #include <bits/stdc++.h>
#endif

#include <random>

static inline int omp_get_thread_num() { return 0; }
static inline int omp_get_max_threads() { return 1; }

using namespace std;

struct Edge { int u, v; double p; };
struct HeapItem {
    double gain;
    int eid;
    int stamp;
};

struct MaxHeapCmp {
    bool operator()(const HeapItem& a, const HeapItem& b) const {
        if (a.gain != b.gain) return a.gain < b.gain;
        if (a.stamp != b.stamp) return a.stamp < b.stamp;
        return a.eid > b.eid;
    }
};

int N = 0, M = 0;
vector<Edge> edges;
vector<vector<int>> adj;
vector<int> seeds;
vector<uint8_t> is_seed;

vector<int> seed_dist;
vector<int> candidates;
vector<int> edge_cand_idx;

vector<uint8_t> blocked;
string output_path;
int k_budget = 0;
int n_sims_param = 0;
int hops_limit = -1;

int W = 200;
int W_words = 0;
// Edge-major live bank: live_edge_words[eid][word] packs 64 worlds.
vector<vector<uint64_t>> live_edge_words;

// Current spread per live graph for unbounded and hop-limited modes.
vector<int> unbounded_spread_state;
vector<long long> block_spread_state;

// Hop-limited state (recomputed when needed).
vector<int> hop_spread_state;

vector<double> last_gain;

struct EvalBuffers {
    vector<uint64_t> vis;
    vector<uint64_t> front;
    vector<int> q;
    vector<int> nq;
    vector<int> touched;
};

auto g_start = chrono::steady_clock::now();

static inline long long elapsed_ms() {
    return chrono::duration_cast<chrono::milliseconds>(
        chrono::steady_clock::now() - g_start).count();
}

static inline uint64_t splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

static inline double u01_from_u64(uint64_t x) {
    return (double)((x >> 11) & ((1ULL << 53) - 1)) * (1.0 / 9007199254740992.0);
}

static inline bool edge_live_in_world(int eid, int w) {
    return ((live_edge_words[eid][w >> 6] >> (w & 63)) & 1ULL) != 0ULL;
}

template <typename Fn>
static inline void for_each_live_world(int eid, const Fn& fn) {
    for (int wd = 0; wd < W_words; wd++) {
        uint64_t bits = live_edge_words[eid][wd];
        while (bits) {
            int b = __builtin_ctzll(bits);
            bits &= (bits - 1);
            int w = (wd << 6) + b;
            if (w < W) fn(w);
        }
    }
}

static inline bool edge_intersects_world_mask(int eid, const vector<uint64_t>& world_mask) {
    for (int wd = 0; wd < W_words; wd++) {
        if (live_edge_words[eid][wd] & world_mask[wd]) return true;
    }
    return false;
}

vector<uint64_t> build_high_damage_world_mask(const vector<int>& spread_state) {
    vector<uint64_t> mask(W_words, 0ULL);
    if ((int)spread_state.size() != W || W == 0) return mask;

    vector<int> ord(W);
    iota(ord.begin(), ord.end(), 0);
    sort(ord.begin(), ord.end(), [&](int a, int b) {
        return spread_state[a] > spread_state[b];
    });

    int top_cnt = max(1, W / 3);
    for (int i = 0; i < top_cnt; i++) {
        int w = ord[i];
        mask[w >> 6] |= (1ULL << (w & 63));
    }
    return mask;
}

vector<uint64_t> build_high_damage_world_mask_from_blocks(const vector<long long>& spread_blocks) {
    vector<uint64_t> mask(W_words, 0ULL);
    if ((int)spread_blocks.size() != W_words || W_words == 0) return mask;

    vector<int> ord(W_words);
    iota(ord.begin(), ord.end(), 0);
    sort(ord.begin(), ord.end(), [&](int a, int b) {
        return spread_blocks[a] > spread_blocks[b];
    });

    int top_blocks = max(1, W_words / 3);
    for (int i = 0; i < top_blocks; i++) {
        int b = ord[i];
        uint64_t m = ~0ULL;
        if (b == W_words - 1 && (W & 63)) m = (1ULL << (W & 63)) - 1ULL;
        mask[b] = m;
    }
    return mask;
}

long long evaluate_64_worlds(int block_idx, int extra_block_eid = -1,
                             const vector<uint8_t>* custom_blocked = nullptr) {
    const vector<uint8_t>& b = custom_blocked ? *custom_blocked : blocked;

    thread_local EvalBuffers buf;
    if ((int)buf.vis.size() != N + 1) {
        buf.vis.assign(N + 1, 0ULL);
        buf.front.assign(N + 1, 0ULL);
    }
    buf.q.clear();
    buf.nq.clear();
    buf.touched.clear();

    auto mark_touched = [&](int x) {
        if (buf.vis[x] == 0ULL && buf.front[x] == 0ULL) buf.touched.push_back(x);
    };

    uint64_t full_mask = ~0ULL;
    if (block_idx == W_words - 1 && (W & 63)) {
        full_mask = (1ULL << (W & 63)) - 1ULL;
    }

    for (int s : seeds) {
        mark_touched(s);
        if (buf.front[s] == 0ULL) buf.q.push_back(s);
        buf.vis[s] |= full_mask;
        buf.front[s] |= full_mask;
    }

    int current_hop = 0;
    while (!buf.q.empty()) {
        if (hops_limit >= 0 && current_hop >= hops_limit) break;

        buf.nq.clear();
        buf.nq.reserve(buf.q.size() * 2 + 8);

        for (int u : buf.q) {
            uint64_t f = buf.front[u];
            buf.front[u] = 0ULL;
            if (!f) continue;

            for (int eid : adj[u]) {
                if (eid == extra_block_eid || b[eid]) continue;

                uint64_t spread = f & live_edge_words[eid][block_idx];
                if (!spread) continue;

                int v = edges[eid].v;
                uint64_t novel = spread & ~buf.vis[v];
                if (!novel) continue;

                mark_touched(v);
                if (buf.front[v] == 0ULL) buf.nq.push_back(v);
                buf.front[v] |= novel;
                buf.vis[v] |= novel;
            }
        }

        buf.q.swap(buf.nq);
        current_hop++;
    }

    long long burned = 0;
    for (int i : buf.touched) burned += __builtin_popcountll(buf.vis[i]);

    for (int i : buf.touched) {
        buf.vis[i] = 0ULL;
        buf.front[i] = 0ULL;
    }

    return burned;
}

void recompute_block_spread_state() {
    block_spread_state.assign(W_words, 0LL);
    for (int b = 0; b < W_words; b++) {
        block_spread_state[b] = evaluate_64_worlds(b);
    }
}

double mean_spread_bitset() {
    long long total = 0;
    for (long long x : block_spread_state) total += x;
    return (double)total / (double)W;
}

void read_graph(const string& path) {
    ifstream fin(path);
    if (!fin) {
        cerr << "Cannot open graph: " << path << '\n';
        exit(1);
    }

    string line;
    while (getline(fin, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
        if (line.empty() || line[0] == '#' || line[0] == '%') continue;
        break;
    }

    istringstream iss(line);
    double a, b, c;
    iss >> a >> b;

    if (iss >> c) {
        edges.push_back({(int)a, (int)b, c});
        N = max(N, max((int)a, (int)b));
        while (getline(fin, line)) {
            while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
            if (line.empty() || line[0] == '#') continue;
            istringstream is2(line);
            int u, v;
            double p;
            if (is2 >> u >> v >> p) {
                edges.push_back({u, v, p});
                N = max(N, max(u, v));
            }
        }
    } else {
        N = (int)a;
        edges.reserve((int)b);
        while (getline(fin, line)) {
            while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
            if (line.empty() || line[0] == '#') continue;
            istringstream is2(line);
            int u, v;
            double p;
            if (is2 >> u >> v >> p) edges.push_back({u, v, p});
        }
    }

    M = (int)edges.size();
    adj.assign(N + 1, {});
    for (int eid = 0; eid < M; eid++) {
        int u = edges[eid].u;
        if (0 <= u && u <= N) adj[u].push_back(eid);
    }
}

void read_seeds(const string& path) {
    ifstream fin(path);
    if (!fin) {
        cerr << "Cannot open seeds: " << path << '\n';
        exit(1);
    }
    string line;
    while (getline(fin, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
        if (line.empty() || line[0] == '#') continue;
        istringstream iss(line);
        int s;
        if (iss >> s && 0 <= s && s <= N) seeds.push_back(s);
    }
}

void write_output(const vector<int>& selected) {
    ofstream out(output_path);
    for (int eid : selected) out << edges[eid].u << ' ' << edges[eid].v << '\n';
}

void compute_distances() {
    const int INF = INT_MAX;
    seed_dist.assign(N + 1, INF);
    queue<int> q;
    for (int s : seeds) {
        if (seed_dist[s] == 0) continue;
        seed_dist[s] = 0;
        q.push(s);
    }
    while (!q.empty()) {
        int u = q.front();
        q.pop();
        for (int eid : adj[u]) {
            int v = edges[eid].v;
            if (seed_dist[v] > seed_dist[u] + 1) {
                seed_dist[v] = seed_dist[u] + 1;
                q.push(v);
            }
        }
    }
}

void find_candidates() {
    candidates.clear();
    edge_cand_idx.assign(M, -1);

    for (int eid = 0; eid < M; eid++) {
        int u = edges[eid].u;
        int v = edges[eid].v;
        if (seed_dist[u] == INT_MAX) continue;
        if (hops_limit >= 0) {
            if (seed_dist[u] >= hops_limit) continue;
            if (seed_dist[v] > hops_limit) continue;
        }
        edge_cand_idx[eid] = (int)candidates.size();
        candidates.push_back(eid);
    }
}

void pick_sample_count() {
    // Dynamically scale statistical robustness based on graph density
    if (M > 1000000) W = 256;       // Massive stress graphs
    else if (M > 200000) W = 512;   // Heavy graphs
    else W = 1024;                  // Standard datasets
}

void sample_live_graphs() {
    W_words = (W + 63) >> 6;
    live_edge_words.assign(M, vector<uint64_t>(W_words, 0ULL));

    for (int eid = 0; eid < M; eid++) {
        auto& words = live_edge_words[eid];
        for (int w = 0; w < W; w++) {
            uint64_t base = splitmix64(0x1234abcdULL ^ (uint64_t)w * 0x9e3779b97f4a7c15ULL);
            uint64_t h = splitmix64(base ^ (uint64_t)(eid + 1) * 0xbf58476d1ce4e5b9ULL);
            double u = u01_from_u64(h);
            if (u <= edges[eid].p) {
                words[w >> 6] |= (1ULL << (w & 63));
            }
        }
    }
}

int spread_hop_live(int w, int extra_block_eid = -1, const vector<uint8_t>* custom_blocked = nullptr) {
    const vector<uint8_t>& b = custom_blocked ? *custom_blocked : blocked;

    vector<int> dist(N + 1, -1);
    vector<int> q;
    q.reserve(N + 1);

    int cnt = 0;
    for (int s : seeds) {
        if (dist[s] == -1) {
            dist[s] = 0;
            q.push_back(s);
            cnt++;
        }
    }

    for (int qi = 0; qi < (int)q.size(); qi++) {
        int u = q[qi];
        if (dist[u] >= hops_limit) continue;

        for (int eid : adj[u]) {
            if (eid == extra_block_eid) continue;
            if (b[eid]) continue;
            if (!edge_live_in_world(eid, w)) continue;

            int v = edges[eid].v;
            int nd = dist[u] + 1;
            if (nd > hops_limit) continue;

            if (dist[v] == -1) {
                dist[v] = nd;
                q.push_back(v);
                cnt++;
            }
        }
    }

    return cnt;
}

int spread_unbounded_live(int w, int extra_block_eid = -1, const vector<uint8_t>* custom_blocked = nullptr) {
    const vector<uint8_t>& b = custom_blocked ? *custom_blocked : blocked;

    vector<uint8_t> vis(N + 1, 0);
    vector<int> q;
    q.reserve(N + 1);

    int cnt = 0;
    for (int s : seeds) {
        if (!vis[s]) {
            vis[s] = 1;
            q.push_back(s);
            cnt++;
        }
    }

    for (int qi = 0; qi < (int)q.size(); qi++) {
        int u = q[qi];
        for (int eid : adj[u]) {
            if (eid == extra_block_eid) continue;
            if (b[eid]) continue;
            if (!edge_live_in_world(eid, w)) continue;
            int v = edges[eid].v;
            if (!vis[v]) {
                vis[v] = 1;
                q.push_back(v);
                cnt++;
            }
        }
    }

    return cnt;
}

void recompute_hop_spread_state() {
    hop_spread_state.assign(W, 0);
    for (int w = 0; w < W; w++) {
        hop_spread_state[w] = spread_hop_live(w);
    }
}

double mean_spread_hop() {
    long long sum = 0;
    for (int x : hop_spread_state) sum += x;
    return (double)sum / (double)W;
}

void recompute_unbounded_spread_state() {
    unbounded_spread_state.assign(W, 0);
    for (int w = 0; w < W; w++) {
        unbounded_spread_state[w] = spread_unbounded_live(w);
    }
}

double mean_spread_unbounded() {
    long long sum = 0;
    for (int x : unbounded_spread_state) sum += x;
    return (double)sum / (double)W;
}

double exact_gain_unbounded(int eid) {
    if (blocked[eid]) return 0.0;
    long long saved_total = 0;

    for (int b = 0; b < W_words; b++) {
        long long cur = block_spread_state[b];
        long long nxt = evaluate_64_worlds(b, eid);
        saved_total += (cur - nxt);
    }

    return (double)saved_total / (double)W;
}

double exact_gain_hop(int eid) {
    if (blocked[eid]) return 0.0;
    long long saved_total = 0;

    for (int b = 0; b < W_words; b++) {
        long long cur = block_spread_state[b];
        long long nxt = evaluate_64_worlds(b, eid);
        saved_total += (cur - nxt);
    }

    return (double)saved_total / (double)W;
}

void score_initial_candidates(vector<double>& approx_gain) {
    int C = (int)candidates.size();
    approx_gain.assign(C, 0.0);

    for (int w = 0; w < W; w++) {
        vector<uint8_t> vis(N + 1, 0);
        vector<int> in_count(N + 1, 0);
        vector<int> parent_edge(N + 1, -1);
        vector<int> depth(N + 1, -1);
        vector<int> order;
        order.reserve(N + 1);

        for (int s : seeds) {
            if (!vis[s]) {
                vis[s] = 1;
                depth[s] = 0;
                order.push_back(s);
            }
        }

        for (int qi = 0; qi < (int)order.size(); qi++) {
            int u = order[qi];
            int du = depth[u];
            for (int eid : adj[u]) {
                if (blocked[eid]) continue;
                if (!edge_live_in_world(eid, w)) continue;
                int v = edges[eid].v;

                if (hops_limit >= 0) {
                    if (du >= hops_limit) continue;
                    int nd = du + 1;
                    if (nd > hops_limit) continue;
                    if (!vis[v]) {
                        vis[v] = 1;
                        depth[v] = nd;
                        parent_edge[v] = eid;
                        order.push_back(v);
                    }
                    in_count[v]++;
                } else {
                    if (!vis[v]) {
                        vis[v] = 1;
                        depth[v] = 0;
                        parent_edge[v] = eid;
                        order.push_back(v);
                    }
                    in_count[v]++;
                }
            }
        }

        vector<int> subtree(N + 1, 1);
        for (int i = (int)order.size() - 1; i >= 0; i--) {
            int v = order[i];
            int pe = parent_edge[v];
            if (pe >= 0) {
                int u = edges[pe].u;
                subtree[u] += subtree[v];
                int ci = edge_cand_idx[pe];
                if (ci >= 0 && in_count[v] > 0) {
                    approx_gain[ci] += (double)subtree[v] / (double)in_count[v];
                }
            }
        }
    }

    for (double& x : approx_gain) x /= (double)W;
}

double evaluate_selection_mean_spread(const vector<int>& selected) {
    vector<uint8_t> test_blocked(M, 0);
    for (int eid : selected) test_blocked[eid] = 1;

    long long total = 0;
    for (int b = 0; b < W_words; b++) {
        total += evaluate_64_worlds(b, -1, &test_blocked);
    }

    return (double)total / (double)W;
}

void deterministic_local_search(vector<int>& selected, double& current_spread) {
    if (selected.empty()) return;

    // Give local search enough time to repair greedy mistakes.
    long long budget_ms = 10000;
    long long end_at = elapsed_ms() + budget_ms;

    unordered_set<int> sel_set(selected.begin(), selected.end());
    vector<pair<double, int>> ranked;

    // For hop-limited settings with moderate candidate sets, refresh candidate gains
    // to guide local search using the current blocked configuration.
    vector<double> refreshed_gain;
    bool use_refreshed = (hops_limit >= 0 && (int)candidates.size() <= 10000);
    if (use_refreshed) {
        refreshed_gain.assign(M, 0.0);
        for (int ci = 0; ci < (int)candidates.size(); ci++) {
            int eid = candidates[ci];
            if (sel_set.count(eid)) continue;
            refreshed_gain[eid] = exact_gain_hop(eid);
        }
    }

    for (int eid : candidates) {
        if (sel_set.count(eid)) continue;
        double g = use_refreshed ? refreshed_gain[eid] : last_gain[eid];
        if (g > 0) ranked.push_back({g, eid});
    }
    sort(ranked.begin(), ranked.end(), greater<pair<double, int>>());

    vector<int> pool;
    int pool_size = min((int)ranked.size(), max(400, (int)selected.size() * 8));
    if (pool_size < 2) return;
    for (int i = 0; i < pool_size; i++) pool.push_back(ranked[i].second);

    bool improved = true;
    int k = selected.size();
    
    // Phase 1: 1-opt Hill Climbing
    while (improved && elapsed_ms() < end_at) {
        improved = false;
        for (int i = 0; i < k && !improved; i++) {
            if (elapsed_ms() >= end_at) break;
            
            for (int j = 0; j < pool_size && !improved; j++) {
                if (elapsed_ms() >= end_at) break;
                
                int out_eid = selected[i];
                int in_eid = pool[j];
                
                // check if in_eid is already in selected
                bool dup = false;
                for(int x = 0; x < k; x++) if(selected[x] == in_eid) dup = true;
                if(dup) continue;

                vector<int> trial = selected;
                trial[i] = in_eid;
                
                double sp = evaluate_selection_mean_spread(trial);
                if (sp < current_spread - 1e-6) {
                    current_spread = sp;
                    selected = trial;
                    pool[j] = out_eid; // Put the removed node back into the pool
                    improved = true;
                    // cerr << "1-opt improvement: " << current_spread << " at " << elapsed_ms() << "ms\n";
                }
            }
        }
    }

    // Phase 2: Random 2-opt Hill Climbing if time permits.
    if (k < 2 || pool_size < 2) {
        fill(blocked.begin(), blocked.end(), 0);
        for (int eid : selected) blocked[eid] = 1;

        if (hops_limit >= 0) {
            recompute_hop_spread_state();
        } else {
            recompute_unbounded_spread_state();
        }
        return;
    }

    vector<uint64_t> high_damage_mask = build_high_damage_world_mask_from_blocks(block_spread_state);

    uint64_t rng_state = 1337;
    int step = 0;
    while (elapsed_ms() < end_at) {
        int i1 = (int)(splitmix64(rng_state++) % (uint64_t)k);
        int i2 = (int)(splitmix64(rng_state++) % (uint64_t)(k - 1));
        if (i2 >= i1) i2++;
        
        int j1 = (int)(splitmix64(rng_state++) % (uint64_t)pool_size);
        int j2 = (int)(splitmix64(rng_state++) % (uint64_t)(pool_size - 1));
        if (j2 >= j1) j2++;
        
        int out1 = selected[i1], out2 = selected[i2];
        int in1 = pool[j1], in2 = pool[j2];
        bool dup = false;
        for(int x = 0; x < k; x++) if(selected[x] == in1 || selected[x] == in2) dup = true;
        if(in1 == in2 || dup) continue;

        // Skip swaps that do not touch currently high-damage worlds.
        bool touches_damage = edge_intersects_world_mask(in1, high_damage_mask)
            || edge_intersects_world_mask(in2, high_damage_mask)
            || edge_intersects_world_mask(out1, high_damage_mask)
            || edge_intersects_world_mask(out2, high_damage_mask);
        if (!touches_damage) continue;
        
        vector<int> trial = selected;
        trial[i1] = in1;
        trial[i2] = in2;
        
        double sp = evaluate_selection_mean_spread(trial);
        if (sp < current_spread - 1e-6) {
            current_spread = sp;
            selected = trial;
            pool[j1] = out1;
            pool[j2] = out2;
            // cerr << "2-opt improvement: " << current_spread << " at " << elapsed_ms() << "ms\n";
        }
        step++;
    }

    fill(blocked.begin(), blocked.end(), 0);
    for (int eid : selected) blocked[eid] = 1;

    if (hops_limit >= 0) {
        recompute_hop_spread_state();
    } else {
        recompute_unbounded_spread_state();
    }
}

int main(int argc, char* argv[]) {
    if (argc != 7) {
        cerr << "Usage: " << argv[0]
             << " <graph> <seeds> <output> <k> <n_sims> <hops>\n";
        return 1;
    }

    string graph_path = argv[1];
    string seed_path = argv[2];
    output_path = argv[3];
    k_budget = atoi(argv[4]);
    n_sims_param = atoi(argv[5]);
    hops_limit = atoi(argv[6]);

    // Assignment constraints.
    k_budget = min(50, max(0, k_budget));
    n_sims_param = min(100, max(1, n_sims_param));

    read_graph(graph_path);
    read_seeds(seed_path);

    is_seed.assign(N + 1, 0);
    for (int s : seeds) is_seed[s] = 1;

    blocked.assign(M, 0);
    last_gain.assign(M, 0.0);

    cerr << "Graph: N=" << N << " M=" << M << '\n';
    cerr << "Seeds: " << seeds.size() << " k=" << k_budget << " hops=" << hops_limit << '\n';
    cerr << "Threads: " << omp_get_max_threads() << '\n';

    if (seeds.empty()) {
        vector<int> dummy;
        for (int i = 0; i < min(k_budget, M); i++) dummy.push_back(i);
        write_output(dummy);
        cerr << "No seeds. Wrote dummy output.\n";
        return 0;
    }

    compute_distances();
    find_candidates();

    if (candidates.empty()) {
        vector<int> dummy;
        for (int i = 0; i < min(k_budget, M); i++) dummy.push_back(i);
        write_output(dummy);
        cerr << "No candidates. Wrote dummy output.\n";
        return 0;
    }

    pick_sample_count();
    sample_live_graphs();

    cerr << "Candidates: " << candidates.size() << " | Live-edge samples W=" << W << '\n';

    double sigma_empty = 0.0;
    recompute_block_spread_state();
    sigma_empty = mean_spread_bitset();

    cerr << "Initial spread (SAA): " << sigma_empty << '\n';

    // --- MASSIVE SCALE PRUNING ---
    vector<double> approx_gain;
    score_initial_candidates(approx_gain);

    vector<pair<double, int>> ranked_cands;
    ranked_cands.reserve(candidates.size());
    for (int ci = 0; ci < (int)candidates.size(); ci++) {
        ranked_cands.push_back({approx_gain[ci], candidates[ci]});
    }
    sort(ranked_cands.begin(), ranked_cands.end(), greater<pair<double, int>>());

    int keep_count = 12000;
    if (M > 1000000) keep_count = 1000;
    else if (M > 200000) keep_count = 3000;
    vector<int> pruned_candidates;
    pruned_candidates.reserve(keep_count);
    for (int i = 0; i < keep_count; i++) {
        pruned_candidates.push_back(ranked_cands[i].second);
    }
    candidates = std::move(pruned_candidates);

    edge_cand_idx.assign(M, -1);
    for (int ci = 0; ci < (int)candidates.size(); ci++) {
        edge_cand_idx[candidates[ci]] = ci;
    }
    cerr << "Pruned down to " << candidates.size()
         << " top candidates for exact CELF evaluation.\n";
    // ----------------------------

    // CELF requires true marginal gains at round 0.
    vector<double> init_gain(candidates.size(), 0.0);
    long long r0_start = elapsed_ms();
    for (int ci = 0; ci < (int)candidates.size(); ci++) {

        // --- ROUND 0 KILL SWITCH (Max 10 minutes) ---
        if (elapsed_ms() - r0_start > 600000LL) {
            cerr << "TIME PREEMPT: Round 0 init exceeded 10 mins. Falling back to proxy heuristic for remaining candidates.\n";
            for (int c_rem = ci; c_rem < (int)candidates.size(); c_rem++) {
                // Use the fast tree-based approx_gain we computed earlier
                init_gain[c_rem] = ranked_cands[c_rem].first; 
            }
            break;
        }
        // --------------------------------------------

        int eid = candidates[ci];
        long long saved_total = 0;
        for (int b = 0; b < W_words; b++) {
            long long cur = block_spread_state[b];
            long long nxt = evaluate_64_worlds(b, eid);
            saved_total += (cur - nxt);
        }
        init_gain[ci] = (double)saved_total / (double)W;
    }

    priority_queue<HeapItem, vector<HeapItem>, MaxHeapCmp> pq;
    for (int ci = 0; ci < (int)candidates.size(); ci++) {
        int eid = candidates[ci];
        double g = max(0.0, init_gain[ci]);
        pq.push({g, eid, 0});
        last_gain[eid] = g;
    }

    vector<int> selected;
    selected.reserve(k_budget);

    double current_spread = sigma_empty;

    for (int round = 0; round < k_budget && !pq.empty(); round++) {

        // --- 58 MINUTE HARD STOP (Outer Loop) ---
        if (elapsed_ms() > 3480000LL) {
            cerr << "TIME LIMIT PREEMPT: Reached 58 minutes. Halting greedy search.\n";
            break;
        }
        // ----------------------------------------

        int stamp = round;
        int chosen = -1;
        double chosen_gain = 0.0;

        while (!pq.empty()) {
            HeapItem top = pq.top();
            pq.pop();

            if (blocked[top.eid]) continue;

            if (top.stamp == stamp) {
                chosen = top.eid;
                chosen_gain = max(0.0, top.gain);
                break;
            }

            double g = (hops_limit >= 0) ? exact_gain_hop(top.eid)
                                         : exact_gain_unbounded(top.eid);
            last_gain[top.eid] = g;
            pq.push({g, top.eid, stamp});
        }

        if (chosen < 0) break;

        blocked[chosen] = 1;
        selected.push_back(chosen);

        recompute_block_spread_state();
        current_spread = mean_spread_bitset();

        write_output(selected);

        cerr << "Round " << (round + 1) << '/' << k_budget
             << ": block (" << edges[chosen].u << ',' << edges[chosen].v << ')'
             << " gain=" << chosen_gain
             << " spread=" << current_spread
             << " [" << elapsed_ms() / 1000 << "s]" << '\n';
    }

    // Deterministic refinement on a fixed SAA bank (no re-sampling).
    // deterministic_local_search(selected, current_spread);

    if ((int)selected.size() < k_budget) {
        unordered_set<int> selected_set(selected.begin(), selected.end());
        vector<pair<double, int>> fallback;
        fallback.reserve(candidates.size());

        for (int eid : candidates) {
            if (selected_set.count(eid)) continue;
            double score = edges[eid].p;
            int u = edges[eid].u;
            if (seed_dist[u] != INT_MAX) score /= (1.0 + seed_dist[u]);
            fallback.push_back({score, eid});
        }
        sort(fallback.begin(), fallback.end(), greater<pair<double, int>>());

        for (auto& [s, eid] : fallback) {
            (void)s;
            if ((int)selected.size() >= k_budget) break;
            if (selected_set.count(eid)) continue;
            selected.push_back(eid);
            selected_set.insert(eid);
        }

        for (int eid = 0; eid < M && (int)selected.size() < k_budget; eid++) {
            if (!selected_set.count(eid)) {
                selected.push_back(eid);
                selected_set.insert(eid);
            }
        }
    }

    write_output(selected);

    // Final deterministic SAA score.
    fill(blocked.begin(), blocked.end(), 0);
    for (int eid : selected) blocked[eid] = 1;

    double sigma_R = 0.0;
    recompute_block_spread_state();
    sigma_R = mean_spread_bitset();

    double rho = (sigma_empty > 0.0) ? (sigma_empty - sigma_R) / sigma_empty : 0.0;

    // cerr << "=== Results ===\n";
    // cerr << "sigma(empty)=" << sigma_empty
    //      << " sigma(R)=" << sigma_R
    //      << " rho=" << rho << '\n';
    // cerr << "Blocked " << selected.size() << " edges, "
    //      << elapsed_ms() / 1000 << "s total\n";

    return 0;
}
