"""Comprehensive classification report tests."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)

from evalreport import ClassificationReport


def test_requires_y_true_and_y_pred():
    with pytest.raises(ValueError, match="requires y_true"):
        ClassificationReport(y_true=None, y_pred=[0, 1]).run_all()
    with pytest.raises(ValueError, match="requires y_true"):
        ClassificationReport(y_true=[0, 1], y_pred=None).run_all()


def test_perfect_binary_classification():
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 0, 1]
    out = ClassificationReport(y_true=y_true, y_pred=y_pred).run_all()
    assert out["metrics"]["accuracy"] == 1.0
    assert out["metrics"]["f1_macro"] == 1.0


def test_metrics_match_sklearn_binary_with_prob():
    y_true = [0, 1, 0, 1, 1, 0]
    y_pred = [0, 1, 0, 0, 1, 1]
    y_prob = np.array([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.6, 0.4], [0.2, 0.8], [0.3, 0.7]])
    report = ClassificationReport(y_true=y_true, y_pred=y_pred, y_prob=y_prob)
    report.run_all()
    m = report.metrics
    assert m["accuracy"] == pytest.approx(accuracy_score(y_true, y_pred))
    assert m["f1_macro"] == pytest.approx(f1_score(y_true, y_pred, average="macro", zero_division=0))
    assert m["log_loss"] == pytest.approx(log_loss(y_true, y_prob), rel=1e-5)
    assert "roc_curve" in report.plots
    assert "pr_curve" in report.plots


def test_multiclass_three_classes_metrics_and_cm():
    y_true = [0, 1, 2, 0, 1, 2, 0]
    y_pred = [0, 2, 2, 0, 1, 1, 0]
    report = ClassificationReport(y_true=y_true, y_pred=y_pred)
    report.run_all()
    cm = report.metrics["confusion_matrix"]
    assert len(cm) == 3 and len(cm[0]) == 3
    assert report.metrics["accuracy"] == pytest.approx(accuracy_score(y_true, y_pred))
    assert report.metrics["f1_macro"] == pytest.approx(
        f1_score(y_true, y_pred, average="macro", zero_division=0)
    )
    # No ROC/PR PNGs for multiclass in current implementation
    assert "confusion_matrix" in report.plots
    assert "roc_curve" not in report.plots


def test_multiclass_with_probability_matrix():
    y_true = [0, 1, 2, 0]
    y_pred = [0, 1, 2, 1]
    rng = np.random.default_rng(42)
    y_prob = rng.random((4, 3))
    y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)
    report = ClassificationReport(y_true=y_true, y_pred=y_pred, y_prob=y_prob)
    report.run_all()
    assert report.metrics["log_loss"] == pytest.approx(log_loss(y_true, y_prob), rel=1e-5)
    # Multiclass ROC AUC may be computed
    if report.metrics.get("roc_auc") is not None:
        assert isinstance(report.metrics["roc_auc"], (int, float))


def test_string_labels():
    y_true = ["cat", "dog", "cat", "bird"]
    y_pred = ["cat", "cat", "cat", "bird"]
    report = ClassificationReport(y_true=y_true, y_pred=y_pred)
    report.run_all()
    assert report.metrics["accuracy"] == 0.75
    assert len(report.insights) >= 0


def test_imbalance_insight_triggered():
    # 10 class 0, 1 class 1 -> ratio 10 >= 5
    y_true = [0] * 10 + [1]
    y_pred = [0] * 9 + [1] + [0]  # some errors
    report = ClassificationReport(y_true=y_true, y_pred=y_pred)
    report.run_all()
    assert any("Class imbalance" in ins for ins in report.insights)


def test_explicit_labels_order(tmp_path):
    report = ClassificationReport(
        y_true=[1, 0],
        y_pred=[0, 1],
        labels=[0, 1],
        output_dir=tmp_path,
    )
    report.run_all()
    cm = report.metrics["confusion_matrix"]
    assert cm[0][0] == 0  # true 0 pred 0
    assert cm[1][1] == 0  # true 1 pred 1


def test_binary_1d_prob_vector():
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 1, 0]
    y_prob = [0.2, 0.8, 0.7, 0.3]  # P(class 1)
    report = ClassificationReport(y_true=y_true, y_pred=y_pred, y_prob=y_prob)
    report.run_all()
    assert report.metrics.get("roc_auc") is not None
    assert "roc_curve" in report.plots
