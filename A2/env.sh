#!/bin/bash
# COL761 A2 - Environment Setup

# Python dependencies for Q1
pip install numpy scikit-learn matplotlib

# Compile C++ for Q2 (single-thread build)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/q2"
g++ -O3 -std=c++17 -march=native -flto -o forest_fire forest_fire.cpp 2>/dev/null || \
g++ -O2 -std=c++17 -o forest_fire forest_fire.cpp 2>/dev/null || \
clang++ -O2 -std=c++17 -o forest_fire forest_fire.cpp
