# Q3 Graph Indexing - Complete Testing Guide

## Prerequisites
```bash
cd /Users/abhiramdharme/COL/COL761/A1/q3
```

## Method 1: Using Shell Scripts

### Step 1: Identify Features (gSpan Mining)

**Mutagenicity:**
```bash
./identify.sh q3_datasets/Mutagenicity/graphs.txt features_muta.txt
```

**NCI-H23:**
```bash
./identify.sh q3_datasets/NCI-H23/graphs.txt features_nci.txt
```

### Step 2: Convert to Pickle Format
```bash
# For Mutagenicity
python3 txt_to_pkl.py features_muta.txt features_muta.pkl

# For NCI-H23
python3 txt_to_pkl.py features_nci.txt features_nci.pkl
```

### Step 3: Convert Database to Feature Vectors

**Mutagenicity:**
```bash
./convert.sh q3_datasets/Mutagenicity/graphs.txt features_muta.pkl db_features_muta.npy
```

**NCI-H23:**
```bash
./convert.sh q3_datasets/NCI-H23/graphs.txt features_nci.pkl db_features_nci.npy
```

### Step 4: Convert Queries to Feature Vectors

**Mutagenicity:**
```bash
./convert.sh test_queries_new.txt features_muta.pkl test_query_features_muta.npy
```

**NCI-H23:**
```bash
./convert.sh test_queries_nci.txt features_nci.pkl test_query_features_nci.npy
```

### Step 5: Generate Candidates

**Note:** `generate_candidates.sh` has a bug (calls filter.py instead of candidate_gen.py). Use direct Python:

**Mutagenicity:**
```bash
python3 candidate_gen.py db_features_muta.npy test_query_features_muta.npy candidates_muta.dat
```

**NCI-H23:**
```bash
python3 candidate_gen.py db_features_nci.npy test_query_features_nci.npy candidates_nci.dat
```

### Step 6: Evaluate

**Mutagenicity:**
```bash
python3 eval.py test_truth_new.dat candidates_muta.dat
```

**NCI-H23:**
```bash
python3 eval.py test_truth_nci.dat candidates_nci.dat
```

---

## Method 2: Direct Python (Recommended)

### Complete Pipeline for Mutagenicity:
```bash
cd /Users/abhiramdharme/COL/COL761/A1/q3

# Step 1: Mine features
python3 identify.py q3_datasets/Mutagenicity/graphs.txt features_muta.txt

# Step 2: Convert to pickle
python3 txt_to_pkl.py features_muta.txt features_muta.pkl

# Step 3: Database vectors
python3 convert.py q3_datasets/Mutagenicity/graphs.txt features_muta.pkl db_features_muta.npy

# Step 4: Query vectors  
python3 convert.py test_queries_new.txt features_muta.pkl test_query_features_muta.npy

# Step 5: Generate candidates
python3 candidate_gen.py db_features_muta.npy test_query_features_muta.npy candidates_muta.dat

# Step 6: Evaluate
python3 eval.py test_truth_new.dat candidates_muta.dat
```

### Complete Pipeline for NCI-H23:
```bash
cd /Users/abhiramdharme/COL/COL761/A1/q3

# Step 1: Mine features
python3 identify.py q3_datasets/NCI-H23/graphs.txt features_nci.txt

# Step 2: Convert to pickle
python3 txt_to_pkl.py features_nci.txt features_nci.pkl

# Step 3: Database vectors
python3 convert.py q3_datasets/NCI-H23/graphs.txt features_nci.pkl db_features_nci.npy

# Step 4: Query vectors
python3 convert.py test_queries_nci.txt features_nci.pkl test_query_features_nci.npy

# Step 5: Generate candidates
python3 candidate_gen.py db_features_nci.npy test_query_features_nci.npy candidates_nci.dat

# Step 6: Evaluate
python3 eval.py test_truth_nci.dat candidates_nci.dat
```

---

## One-Liner Pipelines

### Mutagenicity (all steps):
```bash
python3 identify.py q3_datasets/Mutagenicity/graphs.txt features_muta.txt && \
python3 txt_to_pkl.py features_muta.txt features_muta.pkl && \
python3 convert.py q3_datasets/Mutagenicity/graphs.txt features_muta.pkl db_features_muta.npy && \
python3 convert.py test_queries_new.txt features_muta.pkl test_query_features_muta.npy && \
python3 candidate_gen.py db_features_muta.npy test_query_features_muta.npy candidates_muta.dat && \
python3 eval.py test_truth_new.dat candidates_muta.dat
```

### NCI-H23 (all steps):
```bash
python3 identify.py q3_datasets/NCI-H23/graphs.txt features_nci.txt && \
python3 txt_to_pkl.py features_nci.txt features_nci.pkl && \
python3 convert.py q3_datasets/NCI-H23/graphs.txt features_nci.pkl db_features_nci.npy && \
python3 convert.py test_queries_nci.txt features_nci.pkl test_query_features_nci.npy && \
python3 candidate_gen.py db_features_nci.npy test_query_features_nci.npy candidates_nci.dat && \
python3 eval.py test_truth_nci.dat candidates_nci.dat
```

---

## Expected Results

**Current Performance:**
- **Mutagenicity**: ~0.305 Average Precision
- **NCI-H23**: ~0.271 Average Precision
- **Zero false negatives** (all true matches are in candidates)

---

## Troubleshooting

### If identify.py is slow:
- It's running gSpan mining which can take 1-5 minutes depending on support level
- Watch for progress output showing each support level tried

### If convert.py is slow:
- Uses NetworkX monomorphism checking (slower than rustworkx but more compatible)
- Shows progress every 50 graphs

### If results differ:
- Make sure you're using the correct truth files (`test_truth_new.dat` for Muta, `test_truth_nci.dat` for NCI)
- Check that features were generated successfully (features_*.txt should have exactly 50 patterns)
