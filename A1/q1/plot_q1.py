#!/usr/bin/env python3
"""
Generate runtime comparison plot for Apriori and FP-Growth algorithms.
Usage: python3 plot_q1.py <thresholds> <apriori_times> <fp_times> <output_path>
"""

import sys
import matplotlib.pyplot as plt


def main():
    if len(sys.argv) != 5:
        print("Usage: python3 plot_q1.py <thresholds> <apriori_times> <fp_times> <output_path>")
        sys.exit(1)

    # Parse arguments
    thresholds = [int(x) for x in sys.argv[1].split(',')]
    apriori_times = [float(x) for x in sys.argv[2].split(',')]
    fp_times = [float(x) for x in sys.argv[3].split(',')]
    output_path = sys.argv[4]

    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Apriori: blue solid line with circle markers
    plt.plot(thresholds, apriori_times, 'o-', label='Apriori', color='#1f77b4', linewidth=2, markersize=8)
    # FP-Growth: orange dashed line with square markers
    plt.plot(thresholds, fp_times, 's--', label='FP-Growth', color='#ff7f0e', linewidth=2, markersize=8)
    
    plt.xlabel('Threshold (%)', fontsize=12)
    plt.ylabel('Execution Time (seconds)', fontsize=12)
    plt.title('Support Threshold (in %) vs. Run Time (in s)', fontsize=14)
    plt.legend(fontsize=11, loc='upper right')
    plt.grid(True, alpha=0.5)
    
    # Set x-axis ticks to match thresholds
    plt.xticks(thresholds)
    
    # Ensure y-axis starts at 0
    plt.ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"Plot saved to {output_path}")


if __name__ == "__main__":
    main()
