# Q3 Graph Indexing - Complete Start-to-End Guide

## Overview
The pipeline has 4 main steps:
1. **identify.py** - Mine discriminative features using gSpan
2. **convert.py** - Convert graphs to feature vectors (uses monomorphism)
3. **filter.py** - Generate candidate sets using bitwise filtering
4. **eval.py** - Evaluate precision

---

## Complete Pipeline Commands

### Mutagenicity Dataset

```bash
cd /Users/abhiramdharme/COL/COL761/A1/q3

# Step 1: Mine features (takes 1-3 minutes)
python3 identify.py q3_datasets/Mutagenicity/graphs.txt features_muta.txt

# Step 2: Convert database to feature vectors (takes 2-5 minutes)
python3 convert.py q3_datasets/Mutagenicity/graphs.txt features_muta.txt db_features_muta.npy

# Step 3: Convert queries to feature vectors (fast)
python3 convert.py test_queries_new.txt features_muta.txt query_features_muta.npy

# Step 4: Generate candidates using bitwise filtering (fast)
python3 filter.py db_features_muta.npy query_features_muta.npy candidates_muta.dat

# Step 5: Evaluate precision
python3 eval.py test_truth_new.dat candidates_muta.dat
```

### NCI-H23 Dataset

```bash
cd /Users/abhiramdharme/COL/COL761/A1/q3

# Step 1: Mine features (takes 3-7 minutes)
python3 identify.py q3_datasets/NCI-H23/graphs.txt features_nci.txt

# Step 2: Convert database to feature vectors (takes 10-20 minutes)
python3 convert.py q3_datasets/NCI-H23/graphs.txt features_nci.txt db_features_nci.npy

# Step 3: Convert queries to feature vectors (fast)
python3 convert.py test_queries_nci.txt features_nci.txt query_features_nci.npy

# Step 4: Generate candidates using bitwise filtering (fast)
python3 filter.py db_features_nci.npy query_features_nci.npy candidates_nci.dat

# Step 5: Evaluate precision
python3 eval.py test_truth_nci.dat candidates_nci.dat
```

---

## One-Liner Pipelines

### Mutagenicity (Complete)
```bash
python3 identify.py q3_datasets/Mutagenicity/graphs.txt features_muta.txt && \
python3 convert.py q3_datasets/Mutagenicity/graphs.txt features_muta.txt db_features_muta.npy && \
python3 convert.py test_queries_new.txt features_muta.txt query_features_muta.npy && \
python3 filter.py db_features_muta.npy query_features_muta.npy candidates_muta.dat && \
python3 eval.py test_truth_new.dat candidates_muta.dat
```

### NCI-H23 (Complete)
```bash
python3 identify.py q3_datasets/NCI-H23/graphs.txt features_nci.txt && \
python3 convert.py q3_datasets/NCI-H23/graphs.txt features_nci.txt db_features_nci.npy && \
python3 convert.py test_queries_nci.txt features_nci.txt query_features_nci.npy && \
python3 filter.py db_features_nci.npy query_features_nci.npy candidates_nci.dat && \
python3 eval.py test_truth_nci.dat candidates_nci.dat
```

---

## Using Shell Scripts

You can also use the provided shell scripts:

```bash
# Step 1: Identify
./identify.sh q3_datasets/Mutagenicity/graphs.txt features_muta.txt

# Step 2: Convert database
./convert.sh q3_datasets/Mutagenicity/graphs.txt features_muta.txt db_features_muta.npy

# Step 3: Convert queries
./convert.sh test_queries_new.txt features_muta.txt query_features_muta.npy

# Step 4: Generate candidates
./generate_candidates.sh db_features_muta.npy query_features_muta.npy candidates_muta.dat

# Step 5: Evaluate
python3 eval.py test_truth_new.dat candidates_muta.dat
```

---

## Key Implementation Details

### identify.py
- Uses **gSpan** for frequent subgraph mining
- Adaptive support levels: 30%, 25%, 20%, 15%, 10%, 5%
- Selects exactly **50 features** using:
  - Entropy-based scoring (information gain)
  - Size weighting (larger patterns preferred)
  - Correlation filtering (avoid >90% overlap)
  - Novelty bonus (cover new graphs)

### convert.py
- Uses **NetworkX monomorphism** (not rustworkx)
- Checks subgraph monomorphism for each feature
- Outputs binary NumPy matrix (graphs × 50 features)
- Shows progress every 50 graphs

### filter.py
- Uses **bitwise AND** for fast filtering
- Logic: `(DB & Q) == Q` means DB contains all features of Q
- 0-based indexing for both queries and graphs
- Very fast (milliseconds)

### eval.py
- Computes precision: `|true_results| / |candidates|`
- Reports per-query and average precision

---

## Expected Performance

**Current Results:**
- **Mutagenicity**: ~0.30-0.49 Average Precision (varies by feature selection)
- **NCI-H23**: ~0.27-0.37 Average Precision
- **Zero false negatives** guaranteed (all true matches in candidates)

---

## File Formats

### Input Graph Format
```
# 0
v 0 1
v 1 2
e 0 1 1
# 1
v 0 3
...
```

### Feature File Format (features_*.txt)
```
t # 0 * 0
v 0 1
v 1 2
e 0 1 1
t # 1 * 0
...
```

### Candidate Output Format (candidates_*.dat)
```
q # 0
c # 5 12 23 45 67
q # 1
c # 3 8 15 22
```

---

## Troubleshooting

### If identify.py is slow
- Normal! gSpan mining takes time
- Watch for "Mining at X% support" messages
- Stops early when 50+ patterns found

### If convert.py is slow
- Normal! Monomorphism checking is expensive
- Shows progress every 50 graphs
- For 4337 graphs: ~2-5 minutes
- For 40k graphs: ~10-20 minutes

### If you get encoding errors
- Make sure all input files are UTF-8
- Check that graph files don't have special characters

### If precision is low
- Try adjusting support levels in identify.py (line 251)
- Current optimal: 30% for NCI, varies for Muta
- Lower support = more patterns = potentially better filtering

---

## Quick Test (Recommended)

Test on Mutagenicity first (smaller, faster):

```bash
cd /Users/abhiramdharme/COL/COL761/A1/q3
python3 identify.py q3_datasets/Mutagenicity/graphs.txt features_test.txt && \
python3 convert.py q3_datasets/Mutagenicity/graphs.txt features_test.txt db_test.npy && \
python3 convert.py test_queries_new.txt features_test.txt query_test.npy && \
python3 filter.py db_test.npy query_test.npy candidates_test.dat && \
python3 eval.py test_truth_new.dat candidates_test.dat
```

This should complete in 3-5 minutes total.
