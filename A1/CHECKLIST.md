# COL761 Assignment 1 Completion Checklist

## 1. Prerequisites (on Baadal/Linux)
- [x] **Libraries**: Apriori, FP-Growth, and Gaston are already compiled in `libs/`.
- [x] **Linux Binaries**: `gSpan` and `fsg` are located in `libs/gSpan6/` and `fsg/Linux/`.
- [x] **Yeast Dataset**: Located and converted in `q2/`.
- [ ] **webdocs.dat**: You still need to find the full ~5GB `webdocs.dat` file. The 30MB one in `libs/` is likely incomplete. Check `/home/datasets/` on Baadal or Moodle.

## 2. Execution (Run these on Baadal)

### Part 1: Frequent Itemset Mining
- [ ] **Task 1 (webdocs)**: 
  ```bash
  cd q1
  bash q1_1.sh ../libs/apriori/src/apriori ../libs/fpgrowth/src/fpgrowth ../libs/webdocs.dat ./output_q1
  ```
- [ ] **Task 2 (Synthetic)**:
  ```bash
  cd q1
  bash q1_2.sh 1000 15000
  ```

### Part 2: Frequent Subgraph Mining
- [ ] **Run Comparison**:
  ```bash
  cd q2
  bash q2.sh ../libs/gSpan6/gSpan ../fsg/Linux/fsg ../libs/gaston-1.1/gaston ../Yeast/167.txt_graph ./output_q2
  ```

## 3. Finishing Up
- [ ] **Verify Plots**: Check `q1/output_q1/plot.png` and `q2/output_q2/plot.png`.
- [ ] **Finalize Reports**: 
  - Ensure `q1/q1_report.md` and `q2/q2_report.md` capture your observations.
  - Convert to PDF:
    ```bash
    pandoc q1/q1_report.md -o q1/q1.pdf
    pandoc q2/q2_report.md -o q2/q2.pdf
    ```
- [ ] **Clean Up**: Remove the `libs/`, `fsg/`, and `Yeast/` directories before submitting.
- [ ] **Submit**: Upload the `A1/` folder to Moodle.

**Note**: Since `gSpan` and `fsg` are Linux-only, you **must** run the benchmark commands on Baadal or a Linux machine. The Gaston conversion and scripts are already prepared.
