#!/usr/bin/env python3
"""
Generate synthetic dataset matching Figure 1 runtime behavior.

FIXED PARAMETERS:
- Transactions: 15,000
- Items: 30 (IDs 0-29)

ITEM GROUPS:
1. Core items (0-4): 5 items at 90% support
2. Plateau items (5-19): 15 items at 70% support, 50% co-occur together
3. Tail items (20-29): 10 items at 5% support

TARGET BEHAVIOR:
- Apriori: High & flat at 5-50%, sharp drop at 90%
- FP-Growth: Relatively flat throughout, then drops at 90%
"""

import sys
import random
from collections import Counter


def generate_dataset(num_items: int, num_transactions: int) -> list:
    """Generate dataset with exact specifications."""
    random.seed(42)
    
    # Dynamic item groups based on num_items (expected 20-40)
    # Core: ~15% of items, Plateau: ~50% of items, Tail: ~35% of items
    num_core = max(3, num_items // 6)           # ~5 items for 30
    num_plateau = max(5, num_items // 2)        # ~15 items for 30
    num_tail = num_items - num_core - num_plateau  # remainder
    
    CORE_ITEMS = list(range(0, num_core))
    PLATEAU_ITEMS = list(range(num_core, num_core + num_plateau))
    TAIL_ITEMS = list(range(num_core + num_plateau, num_items))
    
    # Support probabilities
    CORE_PROB = 0.95      # Each core item appears with 95% probability (>90%)
    PLATEAU_PROB = 0.70   # Each plateau item appears with 70% probability (when not co-occurring)
    PLATEAU_COOCCUR = 1.0   # 100% - ALL plateau items ALWAYS together (perfect co-occurrence)
    TAIL_PROB = 0.07      # Tail items at ~7% (frequent at 5%, drop at 10%)
    
    transactions = []
    seen_transactions = set()
    max_attempts = num_transactions * 100
    attempts = 0
    
    while len(transactions) < num_transactions and attempts < max_attempts:
        attempts += 1
        transaction = []
        
        # Step 1: Add each core item with prob 0.9
        for item in CORE_ITEMS:
            if random.random() < CORE_PROB:
                transaction.append(item)
        
        # Step 2: Plateau items - all appear together 62% of the time
        # Higher co-occurrence keeps 50% flat, lower independent prob creates gap
        if random.random() < 0.62:
            # Add ALL plateau items together (creates dense itemsets)
            transaction.extend(PLATEAU_ITEMS)
        else:
            # 38%: add each plateau item independently with 21% prob
            # Total support per item = 62% + 38%*21% ≈ 70%
            for item in PLATEAU_ITEMS:
                if random.random() < 0.21:
                    transaction.append(item)
        
        # Step 3: Add each tail item with prob 0.05
        for item in TAIL_ITEMS:
            if random.random() < TAIL_PROB:
                transaction.append(item)
        
        # Step 4: If transaction is empty, skip (resample)
        if not transaction:
            continue
        
        # Allow duplicate transactions (common in real datasets)
        transactions.append(sorted(transaction))
        
        if len(transactions) >= num_transactions:
            break
    
    return transactions


def write_dataset(transactions: list, output_path: str):
    """Write transactions to file."""
    with open(output_path, 'w') as f:
        for trans in transactions:
            f.write(' '.join(map(str, trans)) + '\n')


def validate_dataset(transactions: list, num_items: int):
    counts = Counter()
    for txn in transactions:
        for item in txn:
            counts[item] += 1
    
    n = len(transactions)
    
    # Dynamic item groups
    num_core = max(3, num_items // 6)
    num_plateau = max(5, num_items // 2)
    num_tail = num_items - num_core - num_plateau
    
    print(f"\n=== Dataset Validation ===")
    print(f"Transactions: {n}")
    print(f"Avg transaction length: {sum(len(t) for t in transactions) / n:.2f}")
    
    print(f"\n=== Item Support (Actual) ===")
    print(f"Core items (0-{num_core-1}) - target 90%:")
    for item in range(num_core):
        pct = (counts.get(item, 0) / n) * 100
        print(f"  Item {item}: {pct:.1f}%")
    
    print(f"Plateau items ({num_core}-{num_core+num_plateau-1}) - target 70%:")
    plateau_supports = [(counts.get(item, 0) / n) * 100 for item in range(num_core, num_core + num_plateau)]
    print(f"  Range: {min(plateau_supports):.1f}% - {max(plateau_supports):.1f}%")
    print(f"  Average: {sum(plateau_supports)/len(plateau_supports):.1f}%")
    
    print(f"Tail items ({num_core+num_plateau}-{num_items-1}) - target 5%:")
    tail_supports = [(counts.get(item, 0) / n) * 100 for item in range(num_core + num_plateau, num_items)]
    if tail_supports:
        print(f"  Range: {min(tail_supports):.1f}% - {max(tail_supports):.1f}%")
    
    # Count frequent items at each threshold
    freq_5 = sum(1 for c in counts.values() if c >= 0.05 * n)
    freq_10 = sum(1 for c in counts.values() if c >= 0.10 * n)
    freq_25 = sum(1 for c in counts.values() if c >= 0.25 * n)
    freq_50 = sum(1 for c in counts.values() if c >= 0.50 * n)
    freq_90 = sum(1 for c in counts.values() if c >= 0.90 * n)
    
    print(f"\n=== Frequent Items at Each Threshold ===")
    print(f"  5%: {freq_5} items (Core + Plateau)")
    print(f" 10%: {freq_10} items (Core + Plateau)")  
    print(f" 25%: {freq_25} items (Core + Plateau)")
    print(f" 50%: {freq_50} items (Core + Plateau)")
    print(f" 90%: {freq_90} items (Core only)")
    
    print(f"\n=== Expected Runtime Behavior ===")
    print(f"5-50%: {freq_5} frequent items → HIGH & FLAT runtime")
    print(f"  90%: {freq_90} frequent items → SHARP DROP")


def main():
    # Default parameters
    num_items = 30
    num_transactions = 15000
    
    if len(sys.argv) == 3:
        num_items = int(sys.argv[1])
        num_transactions = int(sys.argv[2])
    elif len(sys.argv) != 1:
        print(f"Usage: {sys.argv[0]} <num_items> <num_transactions>")
        print(f"  num_items: 20-40 (default: 30)")
        print(f"  num_transactions: typically 15000")
        sys.exit(1)
    
    print(f"Generating dataset: {num_items} items, {num_transactions} transactions...")
    
    transactions = generate_dataset(num_items, num_transactions)
    
    output_path = "generated_transactions.dat"
    write_dataset(transactions, output_path)
    
    validate_dataset(transactions, num_items)
    
    print(f"\nOutput: {output_path}")


if __name__ == "__main__":
    main()
