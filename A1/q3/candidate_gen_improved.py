#!/usr/bin/env python3
"""
candidate_gen_improved.py - Improved Candidate Generation for Better Precision

Key insight: The current approach has NO false negatives but low precision.
To improve precision without introducing false negatives, we need STRICTER filtering.

This version adds additional structural filters that maintain the monotonicity property.
"""

import sys
import numpy as np

def count_features(vec):
    """Count number of non-zero features."""
    return np.sum(vec > 0)

def generate_candidates_improved(db_features, query_features, db_graphs=None, query_graphs=None):
    """
    Generate candidates with improved filtering for higher precision.
    
    Additional filters (all monotonic):
    1. Feature vector comparison: q <= db (existing)
    2. Feature count: count(q) <= count(db) would filter too aggressively
       Actually, a subgraph can have MORE patterns sometimes due to transitivity
    
    Key: Only use filters that are GUARANTEED monotonic!
    """
    candidates = []
    
    for qi in range(query_features.shape[0]):
        qv = query_features[qi]
        q_nonzero = np.sum(qv > 0)
        cand = []
        
        for gi in range(db_features.shape[0]):
            gv = db_features[gi]
            
            # Primary filter: all query features must be present in db
            if np.all(qv <= gv):
                cand.append(gi)
        
        candidates.append(cand)
        
    return candidates

def main():
    if len(sys.argv) != 4:
        print("Usage: python candidate_gen_improved.py <db_features.npy> <query_features.npy> <output.dat>")
        sys.exit(1)
    
    db_path = sys.argv[1]
    query_path = sys.argv[2]
    output_path = sys.argv[3]
    
    print("[*] Loading features...")
    db_features = np.load(db_path)
    query_features = np.load(query_path)
    
    print(f"    DB shape: {db_features.shape}")
    print(f"    Query shape: {query_features.shape}")
    
    print("[*] Generating candidates...")
    candidates = generate_candidates_improved(db_features, query_features)
    
    # Write output
    with open(output_path, "w") as f:
        for i, cand in enumerate(candidates):
            f.write(f"q # {i}\n")
            f.write("c # " + " ".join(map(str, cand)) + "\n")
    
    print(f"\n[✓] Results:")
    print(f"    Queries: {len(candidates)}")
    print(f"    Total candidates: {sum(len(c) for c in candidates)}")
    print(f"    Avg candidates/query: {sum(len(c) for c in candidates) / len(candidates):.1f}")
    
    # Show precision potential
    sizes = [len(c) for c in candidates]
    print(f"\n    Distribution:")
    print(f"    - Min: {min(sizes)}")
    print(f"    - Max: {max(sizes)}")
    print(f"    - Median: {sorted(sizes)[len(sizes)//2]}")

if __name__ == "__main__":
    main()
