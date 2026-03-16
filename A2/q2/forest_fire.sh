#!/bin/bash
# Usage: bash forest_fire.sh <graph_path> <seed_path> <output_path> <k> <n_sims> <hops>

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BINARY="$SCRIPT_DIR/forest_fire"

# Compile if binary doesn't exist or source is newer
if [ ! -f "$BINARY" ] || [ "$SCRIPT_DIR/forest_fire.cpp" -nt "$BINARY" ]; then
    echo "Compiling forest_fire.cpp..." >&2
    # Single-thread build (no OpenMP).
    g++ -O3 -std=c++17 -march=native -flto -o "$BINARY" "$SCRIPT_DIR/forest_fire.cpp" 2>/dev/null
    if [ $? -ne 0 ]; then
        # Fallback: without -march=native -flto.
        g++ -O3 -std=c++17 -o "$BINARY" "$SCRIPT_DIR/forest_fire.cpp" 2>/dev/null
    fi
    if [ $? -ne 0 ]; then
        echo "Trying clang++ (macOS)..." >&2
        clang++ -O2 -std=c++17 -o "$BINARY" "$SCRIPT_DIR/forest_fire.cpp" 2>/dev/null
    fi
    if [ $? -ne 0 ]; then
        echo "Compilation failed!" >&2
        exit 1
    fi
    echo "Compilation successful." >&2
fi

# Run
"$BINARY" "$1" "$2" "$3" "$4" "$5" "$6"
