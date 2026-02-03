#!/usr/bin/env python3
"""
run_pipeline.py - Complete Graph Indexing Pipeline

Usage: python3 run_pipeline.py <db_file> <query_file> <output_prefix>

This script:
1. Identifies 50 discriminative subgraph features using gSpan
2. Converts BOTH DB and query graphs to feature vectors using SAME extraction
3. Generates candidates using filter q <= db
4. Evaluates precision if truth file exists
"""

import sys
import os
import pickle
import numpy as np
import subprocess
from collections import Counter

def main():
    if len(sys.argv) < 4:
        print("Usage: python3 run_pipeline.py <db_file> <query_file> <output_prefix>")
        print("Example: python3 run_pipeline.py q3_datasets/Mutagenicity/graphs.txt queries.txt muta")
        sys.exit(1)
    
    db_path = sys.argv[1]
    query_path = sys.argv[2]
    prefix = sys.argv[3]
    
    features_file = f"{prefix}_features.pkl"
    db_features_file = f"{prefix}_db_features.npy"
    query_features_file = f"{prefix}_query_features.npy"
    candidates_file = f"{prefix}_candidates.dat"
    
    print("=" * 60)
    print("GRAPH INDEXING PIPELINE")
    print("=" * 60)
    print(f"Database: {db_path}")
    print(f"Queries: {query_path}")
    print(f"Prefix: {prefix}")
    print()
    
    # Step 1: Identify features (if not exists)
    if not os.path.exists(features_file):
        print("[Step 1] Mining discriminative features...")
        subprocess.run([
            sys.executable, "identify.py", db_path, features_file
        ], check=True)
    else:
        print(f"[Step 1] Using existing features: {features_file}")
    
    # Step 2: Convert DB graphs
    if not os.path.exists(db_features_file):
        print("[Step 2] Converting database graphs...")
        subprocess.run([
            sys.executable, "convert.py", db_path, features_file, db_features_file
        ], check=True)
    else:
        print(f"[Step 2] Using existing DB features: {db_features_file}")
    
    # Step 3: Convert query graphs  
    print("[Step 3] Converting query graphs...")
    subprocess.run([
        sys.executable, "convert.py", query_path, features_file, query_features_file
    ], check=True)
    
    # Step 4: Generate candidates
    print("[Step 4] Generating candidates...")
    subprocess.run([
        sys.executable, "candidate_gen.py", db_features_file, query_features_file, candidates_file
    ], check=True)
    
    print()
    print("=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Candidates saved to: {candidates_file}")
    
    # Step 5: Evaluate if truth file exists
    truth_file = f"{prefix}_truth_converted.dat"
    if os.path.exists(truth_file):
        print()
        print("[Step 5] Evaluating against ground truth...")
        subprocess.run([
            sys.executable, "eval.py", truth_file, candidates_file
        ])
    else:
        print(f"\nNote: No truth file found at {truth_file}")

if __name__ == "__main__":
    main()
