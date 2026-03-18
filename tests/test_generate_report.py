"""Tests for evalreport.generate_report entrypoint."""

from __future__ import annotations

import json

import pytest

from evalreport import generate_report


def test_unsupported_task_raises():
    with pytest.raises(ValueError, match="Unsupported task"):
        generate_report(task="clustering", y_true=[0], y_pred=[0], output_path="x.html")


def test_auto_timeseries_unsupported_raises(isolated_reports_cwd):
    with pytest.raises(ValueError, match="Unsupported task"):
        generate_report(
            task="auto",
            y_true=[1.0, 2.0],
            y_pred=[1.0, 2.0],
            timestamps=[1, 2],
        )


def test_auto_regression(isolated_reports_cwd):
    generate_report(
        task="auto",
        y_true=[1.0, 2.0, 3.0],
        y_pred=[1.1, 2.0, 2.9],
    )
    p = isolated_reports_cwd / "reports" / "regression_report.html"
    assert p.exists()
    assert (isolated_reports_cwd / "reports" / "evalreport_plots").is_dir()


def test_auto_classification_int_labels(isolated_reports_cwd):
    generate_report(task="auto", y_true=[0, 1, 0], y_pred=[0, 1, 1])
    assert (isolated_reports_cwd / "reports" / "classification_report.html").exists()


def test_default_json_format(isolated_reports_cwd):
    generate_report(
        task="classification",
        y_true=[0, 1],
        y_pred=[0, 1],
        format="json",
    )
    path = isolated_reports_cwd / "reports" / "classification_report.json"
    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert "metrics" in data and "plots" in data


def test_custom_output_path_plots_colocated(tmp_path):
    sub = tmp_path / "my_reports"
    sub.mkdir()
    out = sub / "eval.html"
    generate_report(task="classification", y_true=[0, 1, 0, 1], y_pred=[0, 1, 1, 0], output_path=str(out))
    assert out.exists()
    plot_dir = sub / "evalreport_plots"
    assert plot_dir.is_dir()
    assert any(plot_dir.glob("*.png"))


def test_regression_multiclass_task_name_still_regression(tmp_path):
    out = tmp_path / "r.html"
    generate_report(task="regression", y_true=[1.0, 2.0], y_pred=[1.0, 2.0], output_path=str(out))
    assert out.exists()


@pytest.mark.parametrize("task_alias", ["binary_classification", "multiclass", "multilabel"])
def test_classification_task_aliases(tmp_path, task_alias):
    out = tmp_path / f"{task_alias}.html"
    generate_report(
        task=task_alias,
        y_true=[0, 1, 0],
        y_pred=[0, 1, 1],
        output_path=str(out),
    )
    assert out.exists()
