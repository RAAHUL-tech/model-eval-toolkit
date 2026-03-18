from evalreport import RegressionReport, generate_report


def test_regression_report_basic_metrics():
    y_true = [1.0, 2.0, 3.0, 4.0]
    y_pred = [1.0, 2.5, 2.5, 4.5]

    report = RegressionReport(y_true=y_true, y_pred=y_pred)
    out = report.run_all()

    assert "metrics" in out
    assert out["metrics"]["mae"] > 0
    assert "rmse" in out["metrics"]
    assert "r2" in out["metrics"]
    assert "residual_plot" in report.plots


def test_generate_report_regression_json_save(tmp_path):
    y_true = [1.0, 2.0, 3.0]
    y_pred = [1.0, 2.0, 2.0]

    out_path = tmp_path / "report.json"
    result = generate_report(task="regression", y_true=y_true, y_pred=y_pred, output_path=str(out_path), format="json")

    assert out_path.exists()
    assert "metrics" in result
    assert "mae" in result["metrics"]

