from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from sklearn.datasets import make_blobs
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

from evalreport import ClusteringReport


def test_clustering_requires_inputs():
    with pytest.raises(ValueError, match="requires X and labels"):
        ClusteringReport(X=None, labels=[0]).run_all()
    with pytest.raises(ValueError, match="requires X and labels"):
        ClusteringReport(X=[[0.0, 0.0]], labels=None).run_all()


def test_clustering_metrics_match_sklearn(tmp_path):
    X, y = make_blobs(
        n_samples=120,
        centers=3,
        cluster_std=0.5,
        n_features=2,
        random_state=42,
    )
    report = ClusteringReport(X=X, labels=y, output_dir=tmp_path)
    report.run_all()

    assert report.metrics["num_clusters"] == 3
    assert report.metrics["silhouette_score"] == pytest.approx(silhouette_score(X, y), rel=1e-6)
    assert report.metrics["davies_bouldin_index"] == pytest.approx(davies_bouldin_score(X, y), rel=1e-6)
    assert report.metrics["calinski_harabasz_score"] == pytest.approx(calinski_harabasz_score(X, y), rel=1e-6)


def test_clustering_plots_created(tmp_path):
    X, y = make_blobs(n_samples=60, centers=2, cluster_std=0.7, n_features=3, random_state=0)
    report = ClusteringReport(X=X, labels=y, output_dir=tmp_path)
    report.run_all()

    assert "cluster_scatter_pca" in report.plots
    assert "cluster_size_distribution" in report.plots
    for p in report.plots.values():
        assert Path(p).exists()

