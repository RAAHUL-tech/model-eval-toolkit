from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from evalreport import TimeSeriesReport


def test_timeseries_requires_inputs():
    with pytest.raises(ValueError, match="requires y_true, y_pred, and timestamps"):
        TimeSeriesReport(y_true=[1.0, 2.0], y_pred=[1.0, 2.0], timestamps=None).run_all()


def test_timeseries_metrics_constant_bias(tmp_path):
    ts = np.arange(20)
    y_true = np.sin(ts / 2.0)
    bias = 0.25
    y_pred = y_true + bias

    report = TimeSeriesReport(y_true=y_true, y_pred=y_pred, timestamps=ts, rolling_window=5, output_dir=tmp_path)
    report.run_all()

    assert report.metrics["mean_forecast_error"] == pytest.approx(bias, rel=1e-6)
    assert report.metrics["mae"] == pytest.approx(abs(bias), rel=1e-6)
    assert report.metrics["rmse"] == pytest.approx(abs(bias), rel=1e-6)
    assert "rolling_rmse_mean" in report.metrics
    assert "rolling_rmse_last" in report.metrics
    assert any("bias" in ins.lower() for ins in report.insights)


def test_timeseries_plots_created(tmp_path):
    ts = np.arange(15)
    y_true = np.cos(ts / 3.0)
    y_pred = y_true + 0.1

    report = TimeSeriesReport(y_true=y_true, y_pred=y_pred, timestamps=ts, rolling_window=4, output_dir=tmp_path)
    report.run_all()

    for key in ("actual_vs_forecast", "residuals_over_time", "rolling_rmse"):
        assert key in report.plots
        assert Path(report.plots[key]).exists()

