from pathlib import Path

from evalreport import ClassificationReport, RegressionReport


def test_html_contains_metric_description_and_plot_images(tmp_path):
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 1, 1]
    report = ClassificationReport(y_true=y_true, y_pred=y_pred)
    report.output_dir = tmp_path
    report.run_all()

    out_path = tmp_path / "report.html"
    report.save(out_path, format="html")

    html = out_path.read_text(encoding="utf-8")
    assert "Overall fraction of correct predictions." in html
    assert "plots-grid" in html
    for _name, p in report.plots.items():
        assert Path(p).exists()
        assert p.replace("\\", "/") in html.replace("\\", "/")


def test_markdown_save_contains_metrics(tmp_path):
    report = RegressionReport(y_true=[1.0, 2.0], y_pred=[1.0, 2.0])
    report.run_all()
    md_path = tmp_path / "r.md"
    report.save(md_path, format="markdown")
    text = md_path.read_text(encoding="utf-8")
    assert "# Evaluation Report" in text
    assert "mae" in text.lower() or "MAE" in text


def test_json_includes_plots_dict(tmp_path):
    y_true = [1.0, 2.0, 3.0, 4.0]
    y_pred = [1.1, 1.9, 3.2, 3.8]

    report = RegressionReport(y_true=y_true, y_pred=y_pred)
    report.run_all()

    out_path = tmp_path / "report.json"
    report.save(out_path, format="json")

    content = out_path.read_text(encoding="utf-8")
    assert "\"plots\"" in content

