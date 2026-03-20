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
        (["a", "b"], ["a", "a"]),  # string labels -> text classification
        ([0, 1], [1, 0]),
    ],
)
def test_infer_classification_non_float(y_true, y_pred):
    task = infer_task(y_true=y_true, y_pred=y_pred)
    if all(isinstance(v, str) for v in y_true) and all(isinstance(v, str) for v in y_pred):
        assert task == "text_classification"
    else:
        assert task == "classification"


def test_infer_classification_integer_targets():
    """Integer dtype targets should still be classification (not regression)."""
    y_true = np.array([0, 1, 2], dtype=np.int64)
    y_pred = np.array([0, 1, 1], dtype=np.int64)
    assert infer_task(y_true=y_true, y_pred=y_pred) == "classification"


def test_infer_defaults_to_classification_without_y():
    assert infer_task() == "classification"


def test_infer_segmentation_masks():
    y_true = [[[0, 1], [1, 0]]]
    y_pred = [[[0, 1], [0, 0]]]
    assert infer_task(y_true=y_true, y_pred=y_pred) == "segmentation"


def test_infer_detection_boxes():
    y_true = [[{"bbox": [0, 0, 10, 10], "label": "obj"}]]
    y_pred = [[{"bbox": [1, 1, 9, 9], "label": "obj", "score": 0.9}]]
    assert infer_task(y_true=y_true, y_pred=y_pred) == "detection"


def test_infer_recommendation_lists():
    y_true = [[1, 2], [3]]
    y_pred = [[1, 4, 2], [3, 2, 1]]
    assert infer_task(y_true=y_true, y_pred=y_pred) == "recommendation"


def test_infer_text_generation_long_sentences():
    y_true = ["the cat sat on the mat", "hello world"]
    y_pred = ["the cat sat on mat", "hello"]
    assert infer_task(y_true=y_true, y_pred=y_pred) == "text_generation"


def test_infer_text_classification_short_labels():
    y_true = ["pos", "neg"]
    y_pred = ["pos", "pos"]
    assert infer_task(y_true=y_true, y_pred=y_pred) == "text_classification"


def test_infer_clustering_from_X_and_discrete_y_pred():
    X = [[0.0, 1.0], [1.0, 0.0], [0.1, 0.2]]
    y_pred = [0, 1, 0]
    assert infer_task(X=X, y_pred=y_pred, y_true=None) == "clustering"
