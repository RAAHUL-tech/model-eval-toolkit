from __future__ import annotations

from pathlib import Path

import pytest

from evalreport import TextClassificationReport, TextGenerationReport


def test_text_classification_is_classification(tmp_path):
    report = TextClassificationReport(
        y_true=["pos", "neg", "pos", "neg"],
        y_pred=["pos", "neg", "neg", "neg"],
    )
    report.output_dir = tmp_path
    out = report.run_all()
    assert "accuracy" in out["metrics"]
    assert "confusion_matrix" in report.plots
    assert Path(report.plots["confusion_matrix"]).exists()


def test_text_generation_basic_metrics_and_plot(tmp_path):
    refs = ["the cat sat on the mat", "hello world"]
    preds = ["the cat sat on mat", "hello"]
    report = TextGenerationReport(references=refs, predictions=preds)
    report.output_dir = tmp_path
    out = report.run_all()
    assert out["metrics"]["num_samples"] == 2
    assert 0.0 <= out["metrics"]["bleu_like"] <= 1.0
    assert 0.0 <= out["metrics"]["rouge_l_f1_like"] <= 1.0
    assert "overlap_distribution" in report.plots
    assert Path(report.plots["overlap_distribution"]).exists()


def test_text_generation_requires_equal_lengths():
    with pytest.raises(ValueError, match="equal length"):
        TextGenerationReport(references=["a"], predictions=["a", "b"]).run_all()

