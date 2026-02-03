#!/bin/bash
# identify.sh - Feature identification script
# Usage: ./identify.sh <database_file> <feature_file>

source .venv/bin/activate
python3 identify.py "$1" "$2"
