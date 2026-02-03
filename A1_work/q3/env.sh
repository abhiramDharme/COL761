#!/bin/bash
# env.sh - Environment setup for graph indexing
source .venv/bin/activate

pip install networkx pandas numpy joblib gspan-mining rustworkx
