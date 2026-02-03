#!/usr/bin/env python3
"""Debug: compare query 0 with a true match (graph 4) and a candidate (graph 0)."""
import numpy as np

db = np.load('db_features_muta.npy')
q = np.load('query_features_muta.npy')

q0 = q[0]
db4 = db[4]  # Should be a true match according to ground truth
db0 = db[0]  # Current candidate

print("Query 0 vs DB graph 4 (TRUE MATCH per ground truth):")
print(f"  Query 0 features: {q0}")
print(f"  DB[4] features:   {db4}")
print(f"  q0 <= db4: {np.all(q0 <= db4)}")
print(f"  Failures: {np.sum(q0 > db4)} features where q0 > db4")
print(f"  Failed indices: {np.where(q0 > db4)[0]}")

print("\nQuery 0 vs DB graph 0 (CANDIDATE):")
print(f"  Query 0 features: {q0}")
print(f"  DB[0] features:   {db0}")
print(f"  q0 <= db0: {np.all(q0 <= db0)}")

# Check if graph 0 is in ground truth for query 0
print("\n\nIs graph 0 in the ground truth for query 0?")
print("Ground truth includes: 4 8 15 23 25... (check manually)")
print("Graph 0 is NOT in the ground truth list, but is a candidate - this is a FALSE POSITIVE")
