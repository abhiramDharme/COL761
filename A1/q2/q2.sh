#!/bin/bash

# Part 2: Run gSpan, FSG, and Gaston algorithms on Yeast dataset
# Usage: bash q2.sh <path_gspan_executable> <path_fsg_executable> <path_gaston_executable> <path_dataset> <path_out>

TIMEOUT_SECONDS=3600
TIMEOUT_MESSAGE=""

# Check arguments
if [ "$#" -ne 5 ]; then
    echo "Usage: bash q2.sh <path_gspan_executable> <path_fsg_executable> <path_gaston_executable> <path_dataset> <path_out>"
    exit 1
fi

GSPAN_EXE="$1"
FSG_EXE="$2"
GASTON_EXE="$3"
DATASET="$4"
OUT_DIR="$5"

# Create output directory if it doesn't exist
mkdir -p "$OUT_DIR"

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

THRESHOLDS=(95 50 25 10 5)

echo "Converting dataset to algorithm-specific formats..."

# Clean up any old converted files to prevent double-pathing errors
rm -f "${SCRIPT_DIR}/yeast_gspan.txt"
rm -f "${SCRIPT_DIR}/yeast_fsg.txt"
rm -f "${SCRIPT_DIR}/yeast_gaston.txt"

python3 "${SCRIPT_DIR}/convert_dataset.py" "$DATASET" "${SCRIPT_DIR}"

# Pre-compute number of graphs (only once!)
NUM_GRAPHS=$(grep -c "^t #" "${SCRIPT_DIR}/yeast_gaston.txt" 2>/dev/null || echo "100")
echo "Total graphs: ${NUM_GRAPHS}"

# Arrays to store runtimes
declare -a GSPAN_TIMES
declare -a FSG_TIMES
declare -a GASTON_TIMES

echo "Running algorithms at different support thresholds..."

for thresh in "${THRESHOLDS[@]}"; do
    echo "Processing support threshold: ${thresh}%"
    
    # Calculate support values once per threshold
    SUPPORT=$(python3 -c "print($thresh / 100.0)")
    ABS_SUPPORT=$(python3 -c "import math; print(max(1, int(math.ceil($NUM_GRAPHS * $thresh / 100.0))))")
    
    # Run gSpan
    echo "  Running gSpan..."
    START_TIME=$(python3 -c "import time; print(time.time())")
    timeout ${TIMEOUT_SECONDS} "$GSPAN_EXE" -f "${SCRIPT_DIR}/yeast_gspan.txt" -s "$SUPPORT" -o 2>/dev/null
    EXIT_CODE=$?
    END_TIME=$(python3 -c "import time; print(time.time())")
    GSPAN_TIME=$(python3 -c "print($END_TIME - $START_TIME)")
    
    # Timeout handling (currently disabled)
    if [ $EXIT_CODE -eq 124 ]; then
        echo "$TIMEOUT_MESSAGE" > "${OUT_DIR}/gspan${thresh}"
        GSPAN_TIMES+=("${TIMEOUT_SECONDS}")
        echo "  gSpan TIMEOUT"
    elif [ $EXIT_CODE -ne 0 ]; then
        > "${OUT_DIR}/gspan${thresh}"
        GSPAN_TIMES+=("$GSPAN_TIME")
        echo "  gSpan KILLED (${GSPAN_TIME}s)"
    else
        mv "${SCRIPT_DIR}/yeast_gspan.txt.fp" "${OUT_DIR}/gspan${thresh}" 2>/dev/null || touch "${OUT_DIR}/gspan${thresh}"
        GSPAN_TIMES+=("$GSPAN_TIME")
        echo "  gSpan completed in ${GSPAN_TIME}s"
    fi
    
    # Run FSG
    echo "  Running FSG..."
    START_TIME=$(python3 -c "import time; print(time.time())")
    # FSG creates yeast_fsg.txt.fp as output file
    timeout ${TIMEOUT_SECONDS} "$FSG_EXE" -s "$thresh" "${SCRIPT_DIR}/yeast_fsg.txt" 2>/dev/null
    EXIT_CODE=$?
    END_TIME=$(python3 -c "import time; print(time.time())")
    FSG_TIME=$(python3 -c "print($END_TIME - $START_TIME)")
    
    # Timeout handling (currently disabled)
    if [ $EXIT_CODE -eq 124 ]; then
        echo "$TIMEOUT_MESSAGE" > "${OUT_DIR}/fsg${thresh}"
        FSG_TIMES+=("${TIMEOUT_SECONDS}")
        echo "  FSG TIMEOUT"
    elif [ $EXIT_CODE -ne 0 ]; then
        > "${OUT_DIR}/fsg${thresh}"
        FSG_TIMES+=("$FSG_TIME")
        echo "  FSG KILLED (${FSG_TIME}s)"
    else
        # Move the generated .fp file to output directory
        # FSG creates yeast_fsg.fp (not yeast_fsg.txt.fp)
        mv "${SCRIPT_DIR}/yeast_fsg.fp" "${OUT_DIR}/fsg${thresh}" 2>/dev/null || touch "${OUT_DIR}/fsg${thresh}"
        FSG_TIMES+=("$FSG_TIME")
        echo "  FSG completed in ${FSG_TIME}s"
    fi
    
    # Run Gaston
    echo "  Running Gaston..."
    START_TIME=$(python3 -c "import time; print(time.time())")
    timeout ${TIMEOUT_SECONDS} "$GASTON_EXE" "$ABS_SUPPORT" "${SCRIPT_DIR}/yeast_gaston.txt" "${OUT_DIR}/gaston${thresh}" 2>/dev/null
    EXIT_CODE=$?
    END_TIME=$(python3 -c "import time; print(time.time())")
    GASTON_TIME=$(python3 -c "print($END_TIME - $START_TIME)")
    
    # Timeout handling (currently disabled)
    if [ $EXIT_CODE -eq 124 ]; then
        echo "$TIMEOUT_MESSAGE" > "${OUT_DIR}/gaston${thresh}"
        GASTON_TIMES+=("${TIMEOUT_SECONDS}")
        echo "  Gaston TIMEOUT"
    elif [ $EXIT_CODE -ne 0 ]; then
        > "${OUT_DIR}/gaston${thresh}"
        GASTON_TIMES+=("$GASTON_TIME")
        echo "  Gaston KILLED (${GASTON_TIME}s)"
    else
        touch "${OUT_DIR}/gaston${thresh}"
        GASTON_TIMES+=("$GASTON_TIME")
        echo "  Gaston completed in ${GASTON_TIME}s"
    fi
done

echo ""
echo "All runs completed. Generating plot..."

# Convert arrays to comma-separated strings for Python
GSPAN_TIMES_STR=$(IFS=,; echo "${GSPAN_TIMES[*]}")
FSG_TIMES_STR=$(IFS=,; echo "${FSG_TIMES[*]}")
GASTON_TIMES_STR=$(IFS=,; echo "${GASTON_TIMES[*]}")
THRESH_STR=$(IFS=,; echo "${THRESHOLDS[*]}")

# Generate the plot using Python
python3 "${SCRIPT_DIR}/plot_q2.py" "$THRESH_STR" "$GSPAN_TIMES_STR" "$FSG_TIMES_STR" "$GASTON_TIMES_STR" "${OUT_DIR}/plot.png"

echo "Done! Output saved to ${OUT_DIR}"
