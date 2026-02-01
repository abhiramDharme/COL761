#!/usr/bin/env python3
"""
Generate runtime comparison plot for gSpan, FSG, and Gaston algorithms.
Usage: python3 plot_q2.py <thresholds> <gspan_times> <fsg_times> <gaston_times> <output_path>
"""

import sys
import matplotlib.pyplot as plt


def main():
    if len(sys.argv) != 6:
        print("Usage: python3 plot_q2.py <thresholds> <gspan_times> <fsg_times> <gaston_times> <output_path>")
        sys.exit(1)

    # Parse arguments
    thresholds = [int(x) for x in sys.argv[1].split(',')]
    gspan_times = [float(x) for x in sys.argv[2].split(',')]
    fsg_times = [float(x) for x in sys.argv[3].split(',')]
    gaston_times = [float(x) for x in sys.argv[4].split(',')]
    output_path = sys.argv[5]

    # Create the plot
    plt.figure(figsize=(10, 6))
    
    plt.plot(thresholds, gspan_times, 'o-', label='gSpan', color='blue', linewidth=2, markersize=8)
    plt.plot(thresholds, fsg_times, 's-', label='FSG', color='red', linewidth=2, markersize=8)
    plt.plot(thresholds, gaston_times, '^-', label='Gaston', color='green', linewidth=2, markersize=8)
    
    plt.xlabel('Minimum Support (%)', fontsize=12)
    plt.ylabel('Runtime (seconds)', fontsize=12)
    plt.title('Runtime Comparison: gSpan vs FSG vs Gaston', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
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
