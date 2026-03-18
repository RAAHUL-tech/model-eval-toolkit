"""Comprehensive regression report tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from evalreport import RegressionReport


def test_requires_y_true_and_y_pred():
    with pytest.raises(ValueError, match="requires y_true"):
        RegressionReport(y_true=None, y_pred=[1.0]).run_all()
    with pytest.raises(ValueError, match="requires y_true"):
        RegressionReport(y_true=[1.0], y_pred=None).run_all()


def test_perfect_predictions():
    y = [1.0, 2.0, 3.0, 4.0]
    report = RegressionReport(y_true=y, y_pred=y)
    report.run_all()
    assert report.metrics["mae"] == 0.0
    assert report.metrics["rmse"] == 0.0
    assert report.metrics["r2"] == 1.0


def test_metrics_match_sklearn():
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.2, 1.8, 3.5, 3.9, 4.7])
    report = RegressionReport(y_true=y_true, y_pred=y_pred)
    report.run_all()
    m = report.metrics
    assert m["mae"] == pytest.approx(mean_absolute_error(y_true, y_pred))
    assert m["mse"] == pytest.approx(mean_squared_error(y_true, y_pred))
    assert m["rmse"] == pytest.approx(np.sqrt(mean_squared_error(y_true, y_pred)))
    assert m["r2"] == pytest.approx(r2_score(y_true, y_pred))


def test_bias_insight_positive():
    y_true = [0.0, 0.0, 0.0]
    y_pred = [2.0, 2.0, 2.0]
    report = RegressionReport(y_true=y_true, y_pred=y_pred)
    report.run_all()
    assert any("positive" in ins.lower() and "bias" in ins.lower() for ins in report.insights)


def test_bias_insight_negative():
    y_true = [5.0, 5.0, 5.0]
    y_pred = [3.0, 3.0, 3.0]
    report = RegressionReport(y_true=y_true, y_pred=y_pred)
    report.run_all()
    assert any("negative" in ins.lower() and "bias" in ins.lower() for ins in report.insights)


def test_plots_written_under_output_dir(tmp_path):
    report = RegressionReport(y_true=[1.0, 2.0, 3.0], y_pred=[1.1, 2.0, 2.9])
    report.output_dir = tmp_path
    report.run_all()
    for key in ("residual_plot", "predicted_vs_actual", "error_distribution"):
        assert key in report.plots
        assert Path(report.plots[key]).exists()


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.UndefinedMetricWarning")
def test_single_sample_edge_case():
    report = RegressionReport(y_true=[3.0], y_pred=[3.5])
    report.run_all()
    assert report.metrics["mae"] == 0.5
    assert "rmse" in report.metrics
