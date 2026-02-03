#!/bin/bash
# convert.sh - Convert graphs to feature vectors
# Usage: ./convert.sh <database_file> <feature_file> <output.npy>

python3 convert.py "$1" "$2" "$3"
