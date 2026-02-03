#!/bin/bash

# Task 1: Run Apriori and FP-Tree algorithms at different support thresholds
# Usage: bash q1_1.sh <path_apriori_executable> <path_fp_executable> <path_dataset> <path_out>

TIMEOUT_SECONDS=3600  # 1 hour timeout per algorithm per threshold
TIMEOUT_MESSAGE=""

# Check arguments
if [ "$#" -ne 4 ]; then
    echo "Usage: bash q1_1.sh <path_apriori_executable> <path_fp_executable> <path_dataset> <path_out>"
    exit 1
fi

APRIORI_EXE="$1"
FP_EXE="$2"
DATASET="$3"
OUT_DIR="$4"

# Create output directory if it doesn't exist
mkdir -p "$OUT_DIR"

# Support thresholds to test
THRESHOLDS=(90 50 25 10 5)

# Arrays to store runtimes
declare -a APRIORI_TIMES
declare -a FP_TIMES

echo "Running Apriori and FP-Growth at different support thresholds..."

for thresh in "${THRESHOLDS[@]}"; do
    echo "Processing support threshold: ${thresh}%"
    
    # Run Apriori with timeout
    echo "  Running Apriori (timeout: ${TIMEOUT_SECONDS}s)..."
    START_TIME=$(python3 -c "import time; print(time.time())")
    timeout ${TIMEOUT_SECONDS} "$APRIORI_EXE" -s"$thresh" "$DATASET" "${OUT_DIR}/ap${thresh}"
    EXIT_CODE=$?
    END_TIME=$(python3 -c "import time; print(time.time())")
    APRIORI_TIME=$(python3 -c "print($END_TIME - $START_TIME)")
    
    if [ $EXIT_CODE -eq 124 ]; then
        # Timeout reached - write message to file
        echo "$TIMEOUT_MESSAGE" > "${OUT_DIR}/ap${thresh}"
        APRIORI_TIMES+=("${TIMEOUT_SECONDS}")
        echo "  Apriori TIMEOUT (recorded as ${TIMEOUT_SECONDS}s)"
    elif [ $EXIT_CODE -ne 0 ]; then
        # Killed (OOM or other error) - empty file
        > "${OUT_DIR}/ap${thresh}"
        APRIORI_TIMES+=("$APRIORI_TIME")
        echo "  Apriori KILLED (empty file, recorded time: ${APRIORI_TIME}s)"
    else
        # Success - ensure file exists
        touch "${OUT_DIR}/ap${thresh}"
        APRIORI_TIMES+=("$APRIORI_TIME")
        echo "  Apriori completed in ${APRIORI_TIME}s"
    fi
    
    # Run FP-Growth with timeout
    echo "  Running FP-Growth (timeout: ${TIMEOUT_SECONDS}s)..."
    START_TIME=$(python3 -c "import time; print(time.time())")
    timeout ${TIMEOUT_SECONDS} "$FP_EXE" -s"$thresh" "$DATASET" "${OUT_DIR}/fp${thresh}"
    EXIT_CODE=$?
    END_TIME=$(python3 -c "import time; print(time.time())")
    FP_TIME=$(python3 -c "print($END_TIME - $START_TIME)")
    
    if [ $EXIT_CODE -eq 124 ]; then
        # Timeout reached - write message to file
        echo "$TIMEOUT_MESSAGE" > "${OUT_DIR}/fp${thresh}"
        FP_TIMES+=("${TIMEOUT_SECONDS}")
        echo "  FP-Growth TIMEOUT (recorded as ${TIMEOUT_SECONDS}s)"
    elif [ $EXIT_CODE -ne 0 ]; then
        # Killed (OOM or other error) - empty file
        > "${OUT_DIR}/fp${thresh}"
        FP_TIMES+=("$FP_TIME")
        echo "  FP-Growth KILLED (empty file, recorded time: ${FP_TIME}s)"
    else
        # Success - ensure file exists
        touch "${OUT_DIR}/fp${thresh}"
        FP_TIMES+=("$FP_TIME")
        echo "  FP-Growth completed in ${FP_TIME}s"
    fi
done

echo ""
echo "All runs completed. Generating plot..."

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Convert arrays to comma-separated strings for Python
APRIORI_TIMES_STR=$(IFS=,; echo "${APRIORI_TIMES[*]}")
FP_TIMES_STR=$(IFS=,; echo "${FP_TIMES[*]}")
THRESH_STR=$(IFS=,; echo "${THRESHOLDS[*]}")

# Generate the plot using Python
python3 "${SCRIPT_DIR}/plot_q1.py" "$THRESH_STR" "$APRIORI_TIMES_STR" "$FP_TIMES_STR" "${OUT_DIR}/plot.png"

echo "Done! Output saved to ${OUT_DIR}"
