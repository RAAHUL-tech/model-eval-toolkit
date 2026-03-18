"""Tests for evalreport.core.task_inference.infer_task."""

from __future__ import annotations

import numpy as np
import pytest

from evalreport.core.task_inference import infer_task


def test_infer_timeseries_when_timestamps():
    assert infer_task(timestamps=[1, 2, 3]) == "timeseries"


def test_infer_clustering_when_embeddings():
    assert infer_task(embeddings=[[0.0, 1.0], [1.0, 0.0]]) == "clustering"


def test_infer_regression_float_targets():
    assert infer_task(y_true=[1.0, 2.0, 3.0], y_pred=[1.1, 1.9, 3.0]) == "regression"


@pytest.mark.parametrize(
    "y_true,y_pred",
    [
        ([0, 1, 2], [0, 1, 1]),
        (["a", "b"], ["a", "a"]),
        ([0, 1], [1, 0]),
    ],
)
def test_infer_classification_non_float(y_true, y_pred):
    assert infer_task(y_true=y_true, y_pred=y_pred) == "classification"


def test_infer_classification_integer_targets():
    """Integer dtype targets should still be classification (not regression)."""
    y_true = np.array([0, 1, 2], dtype=np.int64)
    y_pred = np.array([0, 1, 1], dtype=np.int64)
    assert infer_task(y_true=y_true, y_pred=y_pred) == "classification"


def test_infer_defaults_to_classification_without_y():
    assert infer_task() == "classification"
