
# Q3 Submission - Graph Indexing with Discriminative Subgraphs


## FSM Algorithm Used
**gSpan (graph-based Substructure pattern mining)**
- Implementation: gspan_mining Python package
- Support threshold: Adaptive (30%, 25%, 20%, 15%, 10%, 5%)
- Max vertices per subgraph: 8
- Min vertices per subgraph: 2

## Discriminative Subgraph Selection Strategy

We select exactly k=50 discriminative subgraphs using a multi-criteria scoring approach:

### 1. **Density Filtering**
   - **High-density filter (>55%)**: Removes features present in >55% of graphs (weak filtering power)
   - **Low-density filter (<10%)**: Removes rare features with insufficient coverage

### 2. **Coverage-Based Scoring**
   - Optimal coverage target: 25% of database graphs
   - Score formula: `coverage_score = 1.0 - abs(coverage - 0.25) * 2.0`
   - Wider bonus range: 10-50% coverage gets 1.2x multiplier

### 3. **Pattern Size Weighting**
   - Larger patterns (4-6 edges) receive higher exponential bonuses
   - Size 4-5: exp(size * 0.65-0.7) for balanced specificity
   - Size 6+: exp(size * 0.55) to avoid overly complex patterns

### 4. **Diversity Maximization (Greedy Selection)**
   - Anti-correlation penalty: 0.6 * overlap_ratio
   - Prioritizes features covering NEW graphs not yet covered
   - Combined gain: `(new_coverage + 1) * anti_corr * (size^0.5) * coverage_bonus`

This approach balances filtering power, query coverage, and pattern diversity to achieve:
- **Mutagenicity: 0.5030 precision**
- **NCI-H23: 0.3889 precision**

## How to Run

### Step 1: Feature Identification
```bash
./identify.sh <database_graphs.txt> <features_output.txt>
```
Example:
```bash
./identify.sh q3_datasets/Mutagenicity/graphs.txt features_muta.txt
```

**What it does:**
- Mines frequent subgraphs using gSpan at adaptive support levels
- Selects exactly 50 discriminative subgraphs
- Stores selected features in the specified output file

### Step 2: Graph to Vector Conversion
```bash
./convert.sh <graph_file.txt> <features.txt> <output_vectors.npy>
```
Examples:
```bash
# Convert database graphs
./convert.sh q3_datasets/Mutagenicity/graphs.txt features_muta.txt db_features.npy

# Convert query graphs
./convert.sh query_dataset/muta_final_visible features_muta.txt query_features.npy
```

**What it does:**
- Reads graphs and feature subgraphs
- Uses **non-induced subgraph monomorphism** (NetworkX's `GraphMatcher.subgraph_is_monomorphic()`)
- Generates binary (0/1) vectors of length 50
- Saves vectors as NumPy array (.npy file)

### Step 3: Candidate Generation
```bash
./generate_candidates.sh <db_vectors.npy> <query_vectors.npy> <candidates.dat>
```
Example:
```bash
./generate_candidates.sh db_features.npy query_features.npy candidates.dat
```

**What it does:**
- Performs bitwise AND matching between query and database vectors
- For query vector `[1 0 1 0 1]`, finds all DB vectors matching `[1 X 1 X 1]`
- Outputs 0-based indexed candidates in required format:
  ```
  q # 0
  c # 1 2 3 4
  q # 1
  c # 5 7 9
  ```

## Complete Pipeline Example

```bash
# Mutagenicity dataset
./identify.sh q3_datasets/Mutagenicity/graphs.txt features_muta.txt
./convert.sh q3_datasets/Mutagenicity/graphs.txt features_muta.txt db_features_muta.npy
./convert.sh query_dataset/muta_final_visible features_muta.txt query_features_muta.npy
./generate_candidates.sh db_features_muta.npy query_features_muta.npy candidates_muta.dat

# NCI-H23 dataset
./identify.sh q3_datasets/NCI-H23/graphs.txt features_nci.txt
./convert.sh q3_datasets/NCI-H23/graphs.txt features_nci.txt db_features_nci.npy
./convert.sh query_dataset/nci_final_visible features_nci.txt query_features_nci.npy
./generate_candidates.sh db_features_nci.npy query_features_nci.npy candidates_nci.dat
```

## Dependencies
- Python 3.10+
- NumPy
- NetworkX
- pandas
- gspan_mining (included in src/)


### Non-Induced Subgraph Isomorphism
- Uses `nx.algorithms.isomorphism.GraphMatcher`
- Method: `subgraph_is_monomorphic()` for non-induced matching




## File Structure
```
q3/
├── identify.sh              # Feature identification script
├── convert.sh               # Graph to vector conversion
├── generate_candidates.sh   # Candidate generation
├── README.txt               # This file
├── identify.py              # FSM and feature selection
├── convert.py               # Vector conversion logic
├── filter.py                # Bitwise AND matching
├── utils.py                 # Graph parsing utilities
├── src/                     # Helper modules
│   └── gspan_mining/        # gSpan implementation
└── q3_datasets/             # Dataset graphs (not included in submission)
```

## Performance Characteristics
- **Mutagenicity (4,337 graphs, 50 queries)**: ~2-3 minutes total
- **NCI-H23 (40,353 graphs, 50 queries)**: ~5-20 minutes total
- FSM is the most time-consuming step (70-80% of total time)
- Candidate generation is extremely fast (<1 second) using bitwise operations