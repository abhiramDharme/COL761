#!/bin/bash

# Task 2: Generate synthetic transactional dataset
# Usage: bash q1_2.sh <universal_itemset> <num_transactions>

# Check arguments
if [ "$#" -ne 2 ]; then
    echo "Usage: bash q1_2.sh <universal_itemset> <num_transactions>"
    exit 1
fi

UNIVERSAL_ITEMSET="$1"
NUM_TRANSACTIONS="$2"

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Generating dataset with ${NUM_TRANSACTIONS} transactions from itemset of size ${UNIVERSAL_ITEMSET}..."

# Generate the dataset using Python
python3 "${SCRIPT_DIR}/generate_dataset.py" "$UNIVERSAL_ITEMSET" "$NUM_TRANSACTIONS"

echo "Dataset generated: generated_transactions.dat"
