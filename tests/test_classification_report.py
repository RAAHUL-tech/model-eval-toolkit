import numpy as np

from evalreport import ClassificationReport, generate_report


def test_classification_report_basic_metrics():
    y_true = [0, 1, 0, 1, 1, 0]
    y_pred = [0, 1, 0, 0, 1, 1]
    y_prob = [0.1, 0.9, 0.2, 0.4, 0.8, 0.7]  # prob of class 1

    report = ClassificationReport(y_true=y_true, y_pred=y_pred, y_prob=y_prob)
    out = report.run_all()

    assert "metrics" in out
    assert out["metrics"]["accuracy"] == 4 / 6
    assert "f1_macro" in out["metrics"]
    assert out["metrics"]["confusion_matrix"] is not None
    # Plots dictionary should include at least the confusion matrix figure
    assert "confusion_matrix" in report.plots


def test_generate_report_classification_html_save(tmp_path):
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 1, 1]

    out_path = tmp_path / "report.html"
    result = generate_report(task="classification", y_true=y_true, y_pred=y_pred, output_path=str(out_path))

    assert out_path.exists()
    assert "metrics" in result
    assert "accuracy" in result["metrics"]

