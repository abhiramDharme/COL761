#!/usr/bin/env python3
"""Q1 clustering script for COL761."""

import json
import os
import sys
import os
import numpy as np
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA


STUDENT_ID = "cs5221643"
API_BASE = "http://hulk.cse.iitd.ac.in:3000/dataset"
K_RANGE = range(2, 16)


def load_data_from_api(dataset_num):
    """Load dataset from the IIT network API."""
    import urllib.request
    url = f"{API_BASE}?student_id={STUDENT_ID}&dataset_num={dataset_num}"
    print(f"Fetching dataset {dataset_num} from API {url}...")
    with urllib.request.urlopen(url) as response:
        raw_data = response.read().decode('utf-8')
        data = json.loads(raw_data)
    return np.array(data["X"])


def load_data_from_npy(path):
    """Load dataset from a .npy file."""
    return np.load(path)


def find_optimal_k(X_scaled, k_range=K_RANGE):
    """
    Evaluate KMeans clustering for a range of k values.
    Returns: optimal_k, metrics dict
    """
    silhouette_scores = []
    ch_scores = []
    db_scores = []
    inertias = []
    all_labels = {}

    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=42, max_iter=300)
        labels = km.fit_predict(X_scaled)
        all_labels[k] = labels

        sil = silhouette_score(X_scaled, labels)
        ch = calinski_harabasz_score(X_scaled, labels)
        db = davies_bouldin_score(X_scaled, labels)

        silhouette_scores.append(sil)
        ch_scores.append(ch)
        db_scores.append(db)
        inertias.append(km.inertia_)

    # Optimal k: highest silhouette score
    best_idx = np.argmax(silhouette_scores)
    optimal_k = list(k_range)[best_idx]

    return optimal_k, {
        'silhouette': silhouette_scores,
        'ch': ch_scores,
        'db': db_scores,
        'inertia': inertias,
        'labels': all_labels,
    }


def plot_dataset_row(fig, gs, row_idx, X_scaled, optimal_k, metrics, dataset_label, k_range=K_RANGE):
    """Render one dataset's 6-panel clustering analysis on a given row."""
    k_list = list(k_range)

    # --- 1. Elbow Method (WCSS / Inertia) ---
    ax1 = fig.add_subplot(gs[row_idx, 0])
    ax1.plot(k_list, metrics['inertia'], 'b-o', markersize=5, linewidth=1.5)
    ax1.axvline(x=optimal_k, color='r', linestyle='--', alpha=0.7, label=f'Optimal k={optimal_k}')
    ax1.set_xlabel('Number of Clusters (k)', fontsize=10)
    ax1.set_ylabel('Inertia (WCSS)', fontsize=10)
    ax1.set_title(f'Elbow Method ({dataset_label})', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(k_list)

    # --- 2. Silhouette Score ---
    ax2 = fig.add_subplot(gs[row_idx, 1])
    ax2.plot(k_list, metrics['silhouette'], 'r-o', markersize=5, linewidth=1.5)
    ax2.axvline(x=optimal_k, color='b', linestyle='--', alpha=0.7, label=f'Optimal k={optimal_k}')
    ax2.set_xlabel('Number of Clusters (k)', fontsize=10)
    ax2.set_ylabel('Silhouette Score', fontsize=10)
    ax2.set_title(f'Silhouette Analysis ({dataset_label})', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(k_list)

    # --- 3. Calinski-Harabasz Index ---
    ax3 = fig.add_subplot(gs[row_idx, 2])
    ax3.plot(k_list, metrics['ch'], 'g-o', markersize=5, linewidth=1.5)
    ax3.axvline(x=optimal_k, color='r', linestyle='--', alpha=0.7, label=f'Optimal k={optimal_k}')
    ax3.set_xlabel('Number of Clusters (k)', fontsize=10)
    ax3.set_ylabel('CH Index', fontsize=10)
    ax3.set_title(f'Calinski-Harabasz Index ({dataset_label})', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(k_list)

    # --- 4. Silhouette Diagram for optimal k ---
    ax4 = fig.add_subplot(gs[row_idx, 3])
    labels = metrics['labels'][optimal_k]
    sample_silhouette_values = silhouette_samples(X_scaled, labels)
    y_lower = 10
    cmap = plt.colormaps['tab10']
    for i in range(optimal_k):
        ith_cluster_values = sample_silhouette_values[labels == i]
        ith_cluster_values.sort()
        size_cluster_i = ith_cluster_values.shape[0]
        y_upper = y_lower + size_cluster_i
        color = cmap(i / optimal_k)
        ax4.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_values,
                          facecolor=color, edgecolor=color, alpha=0.7)
        ax4.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i), fontsize=9)
        y_lower = y_upper + 10

    avg_sil = metrics['silhouette'][list(k_range).index(optimal_k)]
    ax4.axvline(x=avg_sil, color='red', linestyle='--', label=f'Avg={avg_sil:.3f}')
    ax4.set_xlabel('Silhouette Coefficient', fontsize=10)
    ax4.set_ylabel('Cluster Label', fontsize=10)
    ax4.set_title(f'Silhouette Plot ({dataset_label}, k={optimal_k})', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.set_yticks([])

    # --- 5. PCA 2D Visualization ---
    ax5 = fig.add_subplot(gs[row_idx, 4])
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    scatter = ax5.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab10',
                          s=20, alpha=0.7, edgecolors='k', linewidths=0.3)
    ax5.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=10)
    ax5.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=10)
    ax5.set_title(f'PCA Projection ({dataset_label}, k={optimal_k})', fontsize=12, fontweight='bold')
    plt.colorbar(scatter, ax=ax5, label='Cluster')
    ax5.grid(True, alpha=0.3)

    # --- 6. Davies-Bouldin Index ---
    ax6 = fig.add_subplot(gs[row_idx, 5])
    ax6.plot(k_list, metrics['db'], 'm-o', markersize=5, linewidth=1.5)
    ax6.axvline(x=optimal_k, color='r', linestyle='--', alpha=0.7, label=f'Optimal k={optimal_k}')
    ax6.set_xlabel('Number of Clusters (k)', fontsize=10)
    ax6.set_ylabel('DB Index (lower=better)', fontsize=10)
    ax6.set_title(f'Davies-Bouldin Index ({dataset_label})', fontsize=12, fontweight='bold')
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)
    ax6.set_xticks(k_list)



def generate_plots(dataset_entries, output_path='plot.png', k_range=K_RANGE):
    """Generate one figure containing one row per dataset entry."""
    n_rows = len(dataset_entries)
    fig = plt.figure(figsize=(32, 5 * n_rows))
    gs = fig.add_gridspec(n_rows, 6, hspace=0.5, wspace=0.35)

    title_parts = []
    for row_idx, entry in enumerate(dataset_entries):
        plot_dataset_row(
            fig,
            gs,
            row_idx,
            entry['X_scaled'],
            entry['optimal_k'],
            entry['metrics'],
            entry['label'],
            k_range,
        )
        title_parts.append(f"{entry['label']}: k={entry['optimal_k']}")

    fig.suptitle('Clustering Analysis | ' + ' | '.join(title_parts), fontsize=16, fontweight='bold', y=0.995)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 Q1.py <dataset_num> or python3 Q1.py <path_to_dataset.npy>",
              file=sys.stderr)
        sys.exit(1)

    arg = sys.argv[1]

    dataset_entries = []

    # Determine input mode
    if arg.endswith('.npy'):
        data = load_data_from_npy(arg)
        dataset_label = os.path.basename(arg)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(data)
        optimal_k, metrics = find_optimal_k(X_scaled)
        dataset_entries.append({
            'label': dataset_label,
            'X_scaled': X_scaled,
            'optimal_k': optimal_k,
            'metrics': metrics,
        })
    elif arg.endswith('.json'):
        # Convenience: also handle .json files
        with open(arg) as f:
            raw = json.load(f)
        data = np.array(raw["X"])
        dataset_label = os.path.basename(arg)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(data)
        optimal_k, metrics = find_optimal_k(X_scaled)
        dataset_entries.append({
            'label': dataset_label,
            'X_scaled': X_scaled,
            'optimal_k': optimal_k,
            'metrics': metrics,
        })
    else:
        try:
            int(arg)
            # Numeric input triggers both official API datasets in one figure.
            for dataset_num in (1, 2):
                data = load_data_from_api(dataset_num)
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(data)
                optimal_k, metrics = find_optimal_k(X_scaled)
                dataset_entries.append({
                    'label': f"Dataset {dataset_num}",
                    'X_scaled': X_scaled,
                    'optimal_k': optimal_k,
                    'metrics': metrics,
                })
        except ValueError:
            print(f"Invalid argument: {arg}", file=sys.stderr)
            sys.exit(1)

    generate_plots(dataset_entries)

    if len(dataset_entries) == 1:
        print(dataset_entries[0]['optimal_k'])
    else:
        for entry in dataset_entries:
            print(f"{entry['label']}: {entry['optimal_k']}")


if __name__ == "__main__":
    main()
