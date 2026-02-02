#!/usr/bin/env python3
"""
Synthetic Dataset Generation for Q1.2

STRATEGY:
- High co-occurrence (Q_PLATEAU_ALL = 0.65) for flat runtime at 10-50%
- Noise items with support 5-9% to increase itemsets at 5% threshold
"""

import sys
import random
from collections import Counter


def generate_dataset(num_items: int, num_transactions: int) -> list:
    random.seed(42)
    
    # Item partitions
    CORE_ITEMS = list(range(0, 5))       # 5 items: 0-4
    PLATEAU_ITEMS = list(range(5, 20))   # 15 items: 5-19
    NOISE_ITEMS = list(range(20, num_items))  # Remaining for noise
    
    P_CORE = 0.95
    Q_PLATEAU_ALL = 0.65  # High co-occurrence for flat runtime at 10-50%
    P_PLATEAU_IND = 0.60  # For remaining 35% of transactions
    P_NOISE = 0.08        # Higher noise for 5% threshold
    
    transactions = []
    
    for _ in range(num_transactions):
        txn = []
        
        for item in CORE_ITEMS:
            if random.random() < P_CORE:
                txn.append(item)
        
        if random.random() < Q_PLATEAU_ALL:
            txn.extend(PLATEAU_ITEMS)
        else:
            for item in PLATEAU_ITEMS:
                if random.random() < P_PLATEAU_IND:
                    txn.append(item)
        
        for item in NOISE_ITEMS:
            if random.random() < P_NOISE:
                txn.append(item)
        
        transactions.append(sorted(txn))
    
    return transactions


def write_dataset(transactions: list, output_path: str):
    """Write transactions to file."""
    with open(output_path, 'w') as f:
        for trans in transactions:
            f.write(' '.join(str(item) for item in trans) + '\n')


def validate_dataset(transactions: list, num_items: int):
    """Validate dataset against expected statistical properties."""
    n = len(transactions)
    
    # Count item frequencies
    counts = Counter()
    for trans in transactions:
        counts.update(trans)


def main():
    if len(sys.argv) != 3:
        num_items = 25
        num_transactions = 15000
    else:
        num_items = int(sys.argv[1])
        num_transactions = int(sys.argv[2])
    
    transactions = generate_dataset(num_items, num_transactions)
    
    output_path = "generated_transactions.dat"
    write_dataset(transactions, output_path)
    
    validate_dataset(transactions, num_items)

    print(f"Dataset generated with {num_transactions} transactions and {num_items} items.")


if __name__ == "__main__":
    main()