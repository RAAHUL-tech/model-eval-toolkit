from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from evalreport import ImageClassificationReport, generate_report


def test_image_classification_report_binary_plots(tmp_path):
    y_true = [0, 1, 0, 1, 0, 1]
    y_pred = [0, 1, 0, 0, 1, 1]
    # P(class 1)
    y_prob = np.array([0.1, 0.9, 0.2, 0.4, 0.8, 0.7], dtype=float)

    report = ImageClassificationReport(y_true=y_true, y_pred=y_pred, y_prob=y_prob)
    report.output_dir = tmp_path
    out = report.run_all()

    assert out["metrics"]["accuracy"] >= 0.0
    assert "confusion_matrix" in report.plots
    assert "roc_curve" in report.plots
    assert "pr_curve" in report.plots

    for _, p in report.plots.items():
        assert Path(p).exists()


def test_generate_report_image_classification_html_and_plots(tmp_path):
    out_path = tmp_path / "img_cls.html"
    result = generate_report(
        task="image_classification",
        y_true=[0, 1, 0, 1],
        y_pred=[0, 1, 1, 1],
        y_prob=[0.2, 0.8, 0.7, 0.9],
        output_path=str(out_path),
        format="html",
    )

    assert out_path.exists()
    assert "metrics" in result

    plot_dir = tmp_path / "evalreport_plots"
    assert plot_dir.is_dir()
    # Confusion matrix should always be present
    assert any(plot_dir.glob("*confusion_matrix*.png"))


def test_image_classification_multiclass_confusion_matrix_only(tmp_path):
    # Multiclass: ROC/PR curve PNGs are binary-only in v0.1
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 2, 2, 0, 1, 1]
    report = ImageClassificationReport(y_true=y_true, y_pred=y_pred)
    report.output_dir = tmp_path
    report.run_all()

    assert "confusion_matrix" in report.plots
    assert "roc_curve" not in report.plots
    assert "pr_curve" not in report.plots

