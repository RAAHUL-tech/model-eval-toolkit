from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

from ..core.base_report import BaseReport


def _as_2d_array(x: Optional[Iterable[Any]]) -> Optional[np.ndarray]:
    if x is None:
        return None
    arr = np.asarray(list(x))
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def _as_array(x: Optional[Iterable[Any]]) -> Optional[np.ndarray]:
    if x is None:
        return None
    return np.asarray(list(x))


@dataclass
class ClusteringReport(BaseReport):
    X: Optional[Iterable[Any]] = None
    labels: Optional[Iterable[Any]] = None
    # When provided, we will fit a basic KMeans model to derive cluster centers
    # for plotting/diagnostics.
    n_clusters: Optional[int] = None
    random_state: int = 0

    def _compute_metrics(self) -> None:
        X = _as_2d_array(self.X)
        labels = _as_array(self.labels)
        if X is None or labels is None:
            raise ValueError("ClusteringReport requires X and labels (cluster assignments).")

        unique = np.unique(labels)
        if unique.size < 2:
            self.metrics.update(
                {
                    "silhouette_score": None,
                    "davies_bouldin_index": None,
                    "calinski_harabasz_score": None,
                }
            )
        else:
            self.metrics["silhouette_score"] = float(silhouette_score(X, labels))
            self.metrics["davies_bouldin_index"] = float(davies_bouldin_score(X, labels))
            self.metrics["calinski_harabasz_score"] = float(calinski_harabasz_score(X, labels))

        self.metrics["num_clusters"] = int(unique.size)
        # cluster_size distribution as counts
        vals, counts = np.unique(labels, return_counts=True)
        self.metrics["cluster_sizes"] = {str(v): int(c) for v, c in zip(vals, counts)}

        # Human-readable explanations
        self.metric_descriptions.update(
            {
                "silhouette_score": "How well points fit their own cluster vs other clusters (higher is better).",
                "davies_bouldin_index": "Average similarity between clusters (lower is better).",
                "calinski_harabasz_score": "Variance ratio criterion (higher suggests clearer separation).",
                "num_clusters": "Number of unique clusters in the provided assignments.",
                "cluster_sizes": "Counts per cluster; helps detect cluster imbalance.",
            }
        )

    def _generate_plots(self) -> None:
        X = _as_2d_array(self.X)
        labels = _as_array(self.labels)
        if X is None or labels is None:
            self.plots = {}
            return

        root = self.output_dir or Path("reports")
        plot_dir = root / "evalreport_plots"
        plot_dir.mkdir(parents=True, exist_ok=True)

        plots: dict[str, str] = {}

        # Scatter (PCA projection to 2D)
        try:
            pca_dim = 2 if X.shape[1] >= 2 and X.shape[0] >= 3 else 1
            pca_dim = min(pca_dim, X.shape[1], max(1, X.shape[0] - 1))
            pca = PCA(n_components=pca_dim, random_state=self.random_state)
            X2 = pca.fit_transform(X)

            plt.figure(figsize=(5, 4))
            if X2.shape[1] == 1:
                plt.scatter(X2[:, 0], np.zeros_like(X2[:, 0]), c=labels, cmap="tab10", alpha=0.8)
            else:
                plt.scatter(X2[:, 0], X2[:, 1], c=labels, cmap="tab10", alpha=0.8)
            plt.title("Cluster scatter (PCA)")
            plt.xlabel("PC1")
            plt.ylabel("PC2" if X2.shape[1] > 1 else "")
            path = plot_dir / "clustering_scatter_pca.png"
            plt.tight_layout()
            plt.savefig(path)
            plt.close()
            plots["cluster_scatter_pca"] = str(path)
        except Exception:
            pass

        # Cluster size distribution
        try:
            vals, counts = np.unique(labels, return_counts=True)
            plt.figure(figsize=(5, 3.5))
            sns.barplot(x=[str(v) for v in vals], y=counts, color="#4C78A8")
            plt.xlabel("Cluster")
            plt.ylabel("Count")
            plt.title("Cluster size distribution")
            for i, c in enumerate(counts):
                plt.text(i, c, str(int(c)), ha="center", va="bottom", fontsize=8)
            path = plot_dir / "clustering_cluster_sizes.png"
            plt.tight_layout()
            plt.savefig(path)
            plt.close()
            plots["cluster_size_distribution"] = str(path)
        except Exception:
            pass

        self.plots = plots

    def _generate_insights(self) -> None:
        labels = _as_array(self.labels)
        if labels is None:
            self.insights = []
            return

        insights: List[str] = []
        unique, counts = np.unique(labels, return_counts=True)
        if unique.size >= 2:
            max_c = counts.max()
            min_c = counts.min()
            ratio = float(max_c) / float(max(1, min_c))
            if ratio >= 5:
                # report most/least dominant clusters
                maj = unique[np.argmax(counts)]
                min_label = unique[np.argmin(counts)]
                insights.append(f"Cluster imbalance detected (majority={maj!r}, minority={min_label!r}, ratio≈{ratio:.1f}).")

        # Separability heuristics using silhouette
        sil = self.metrics.get("silhouette_score")
        if isinstance(sil, (int, float)) and sil is not None:
            if sil < 0.25:
                insights.append("Clusters overlap significantly (low silhouette). Consider revisiting features, scaling, or k.")
            elif sil > 0.5:
                insights.append("Clusters appear well separated (high silhouette).")

        # Davies-Bouldin lower is better
        dbi = self.metrics.get("davies_bouldin_index")
        if isinstance(dbi, (int, float)) and dbi is not None:
            if dbi > 1.0:
                insights.append("Higher Davies–Bouldin suggests clusters may be less distinct; inspect overlaps.")

        self.insights = insights

