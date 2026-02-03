#!/bin/bash
# env.sh - Environment setup for graph indexing
python3 -m venv .venv
source .venv/bin/activate

pip install networkx pandas numpy joblib gspan-mining rustworkx
