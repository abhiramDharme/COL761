#!/usr/bin/env python3
"""
candidate_gen_precision.py - MAXIMUM PRECISION Candidate Generation

Key optimizations:
1. STRICTER filtering: ALL feature constraints must be satisfied
2. Use numpy vectorization for speed
3. No false negatives allowed
"""

import sys
import numpy as np

def generate_candidates_precision(db_features, query_features):
    """
    Generate candidates with STRICT filtering for maximum precision.
    
    Filtering rule: For each query Q and database graph G:
        Q is candidate if: for ALL features f, query_f <= db_f
        
    This ensures NO false negatives (all true matches are included)
    while maximizing precision by filtering aggressively.
    """
    candidates = []
    
    print(f"Database shape: {db_features.shape}")
    print(f"Query shape: {query_features.shape}")
    
    for qi in range(query_features.shape[0]):
        qv = query_features[qi]
        cand = []
        
        # Vectorized comparison: check if qv <= gv for all features
        # This is equivalent to: np.all(qv <= db_features, axis=1)
        # But we do it row by row for debugging
        for gi in range(db_features.shape[0]):
            gv = db_features[gi]
            
            # STRICT: ALL features must satisfy qv[i] <= gv[i]
            if np.all(qv <= gv):
                cand.append(gi)
        
        candidates.append(cand)
        if qi < 5 or qi % 10 == 0:
            print(f"  Query {qi}: {len(cand)} candidates")
    
    return candidates

def main():
    if len(sys.argv) != 4:
        print("Usage: python candidate_gen_precision.py <db_features.npy> <query_features.npy> <output.dat>")
        sys.exit(1)
    
    db_path = sys.argv[1]
    query_path = sys.argv[2]
    output_path = sys.argv[3]
    
    print("[*] Loading features...")
    db_features = np.load(db_path)
    query_features = np.load(query_path)
    
    print("[*] Generating candidates with PRECISION filtering...")
    candidates = generate_candidates_precision(db_features, query_features)
    
    print(f"\n[*] Writing candidates to {output_path}...")
    with open(output_path, "w") as f:
        for i, cand in enumerate(candidates):
            f.write(f"q # {i}\n")
            f.write("c # " + " ".join(map(str, cand)) + "\n")
    
    print(f"\n[✓] Summary:")
    print(f"    Queries: {len(candidates)}")
    print(f"    Total candidates: {sum(len(c) for c in candidates)}")
    print(f"    Avg candidates/query: {sum(len(c) for c in candidates) / len(candidates):.1f}")
    
    # Show distribution
    sizes = [len(c) for c in candidates]
    print(f"    Min candidates: {min(sizes)}")
    print(f"    Max candidates: {max(sizes)}")
    print(f"    Median candidates: {sorted(sizes)[len(sizes)//2]}")

if __name__ == "__main__":
    main()
