# COL761 Data Mining - Assignment 1

### Transferring to Baadal VM
To transfer the project to Baadal and run the Linux-specific benchmarks (gSpan, FSG, Task 1):

1. **On your Mac (Local):**
   ```bash
   rsync -avz --exclude 'libs/apriori/src/apriori' \
               --exclude 'libs/fpgrowth/src/fpgrowth' \
               --exclude 'libs/gaston-1.1/gaston' \
               --exclude 'q1/output/*' \
               --exclude 'q2/output/*' \
               --exclude '*.o' \
               --exclude '.DS_Store' \
               /Users/abhiramdharme/COL/COL761/A1 10.17.6.69:~/
   ```

2. **On Baadal (Remote):**
   ```bash
   ssh baadalvm@10.17.6.69
   cd ~/A1/libs/apriori/src && make clean && make
   cd ~/A1/libs/fpgrowth/src && make clean && make
   cd ~/A1/libs/gaston-1.1 && make clean && make
   ```

## Directory Structure

```
A1/
├── q1/                          # Part 1: Frequent Itemset Mining
│   ├── q1_1.sh                  # Task 1 pipeline script
│   ├── q1_2.sh                  # Task 2 dataset generation script
│   ├── plot_q1.py               # Plotting script
│   ├── generate_dataset.py      # Synthetic dataset generator
│   └── q1_report.md             # Analysis report (convert to PDF)
├── q2/                          # Part 2: Frequent Subgraph Mining
│   ├── q2.sh                    # Main pipeline script
│   ├── convert_dataset.py       # Dataset format converter
│   ├── plot_q2.py               # Plotting script
│   └── q2_report.md             # Analysis report (convert to PDF)
├── libs/                        # External libraries (Apriori, FP-Growth, Gaston, gSpan, FSG)
├── Yeast/                       # Yeast dataset files
├── README.md                    # This file
└── CHECKLIST.md                 # Completion checklist
```

## Setup and Execution

### Prerequisites
- Python 3 with `matplotlib` installed
- GCC/G++ for library compilation

### Part 1: Frequent Itemset Mining
#### Task 1 (webdocs.dat)
1. Run the pipeline script:
   ```bash
   cd q1
   bash q1_1.sh <apriori_path> <fpgrowth_path> <dataset_path> <output_dir>
   ```

#### Task 2 (Synthetic Dataset)
1. Run the generation and benchmark script:
   ```bash
   cd q1
   bash q1_2.sh
   ```

### Part 2: Frequent Subgraph Mining
1. Run the full comparison pipeline:
   ```bash
   cd q2
   bash q2.sh <gspan_exe> <fsg_exe> <gaston_exe> <yeast_data_path> <output_dir>
   ```

## Progress Tracker
- [/] Part 1: Frequent Itemset Mining
    - [x] Compilation (Local)
    - [ ] Obtain full `webdocs.dat` (Server)
    - [ ] Run Task 1 Benchmark on Baadal
    - [x] Task 2 (Synthetic Dataset)
    - [x] Task 1/2 Analysis Report
- [/] Part 2: Frequent Subgraph Mining
    - [x] Compilation (Gaston Local, others on Baadal)
    - [x] Dataset Conversion
    - [x] Gaston Benchmark (Local)
    - [ ] gSpan & FSG Benchmark (Baadal)
    - [x] Analysis Report
