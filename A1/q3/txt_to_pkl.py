#!/usr/bin/env python3
"""Convert text feature file to pickle format for convert.py"""

import sys
import pickle
from common import parse_graph_database

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python txt_to_pkl.py <features.txt> <features.pkl>")
        sys.exit(1)
    
    txt_path = sys.argv[1]
    pkl_path = sys.argv[2]
    
    print(f"Loading features from {txt_path}...")
    features = parse_graph_database(txt_path)
    print(f"Loaded {len(features)} features")
    
    print(f"Saving to {pkl_path}...")
    with open(pkl_path, 'wb') as f:
        pickle.dump(features, f)
    
    print("Done!")
