#!/bin/bash
# generate_candidates.sh - Generate candidate matches
# Usage: ./generate_candidates.sh <db.npy> <query.npy> <candidates.dat>

python3 filter.py "$1" "$2" "$3"
