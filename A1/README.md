# COL761 Data Mining - Assignment 1

## Transferring to Baadal VM

To transfer the project to Baadal and run the Linux-specific benchmarks:

1. **On your Mac (Local):**
   ```bash
   rsync -avz --exclude 'libs/apriori/src/apriori' \
               --exclude 'libs/fpgrowth/src/fpgrowth' \
               --exclude 'libs/gaston-1.1/gaston' \
               --exclude 'q1/output/*' \
               --exclude 'q2/output/*' \
               --exclude '*.o' \

2. **On Baadal (Remote):**
   ```bash
   cd ~/A1/libs/apriori/src && make clean && make
   cd ~/A1/libs/fpgrowth/src && make clean && make
   cd ~/A1/libs/gaston-1.1 && make clean && make
   cd ~/A1/libs/gSpan6 && make clean && make
   cd ~/A1/fsg/Linux && make clean && make
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
├── libs/                        # External libraries
│   ├── apriori/                 # Apriori implementation
│   ├── fpgrowth/                # FP-Growth implementation
│   ├── gaston-1.1/              # Gaston implementation
│   ├── gSpan6/                  # gSpan implementation
│   └── fsg/                     # FSG implementation
├── Yeast/                       # Yeast dataset files
│   └── 167.txt_graph            # Yeast graph dataset
├── README.md                    # This file
└── CHECKLIST.md                 # Completion checklist
```

## Setup and Execution

### Prerequisites
- Python 3 with `matplotlib` installed
- GCC/G++ for library compilation

### Part 1: Frequent Itemset Mining

#### Task 1 (webdocs.dat)
```bash
cd q1
bash q1_1.sh ../libs/apriori/src/apriori ../libs/fpgrowth/src/fpgrowth ../webdocs.dat ./output_q1_1
```

#### Task 2 (Synthetic Dataset)
```bash
cd q1
bash q1_2.sh 40 15000
```

### Part 2: Frequent Subgraph Mining
```bash
cd q2
bash q2.sh ../libs/gSpan6/gSpan ../fsg/Linux/fsg ../libs/gaston-1.1/gaston ../Yeast/167.txt_graph ./output_q2
```

## Progress Tracker

- [x] Part 1: Frequent Itemset Mining
    - [x] Compilation (Local & Baadal)
    - [x] Task 1 Benchmark (webdocs.dat on Baadal)
    - [x] Task 2 Synthetic Dataset Generation
    - [x] Plotting scripts
    - [ ] Task 1/2 Analysis Report
    
- [x] Part 2: Frequent Subgraph Mining
    - [x] Compilation (All algorithms on Baadal)
    - [x] Dataset Conversion (with Piazza label mapping)
    - [x] gSpan, FSG, Gaston Benchmarks
    - [x] Plotting script
    - [ ] Analysis Report
