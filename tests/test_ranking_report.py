from __future__ import annotations

from pathlib import Path

import pytest

from evalreport import RankingReport, generate_report


def test_ranking_requires_same_length():
    with pytest.raises(ValueError, match="same length"):
        RankingReport(relevant=[[1, 2]], ranked=[[1], [2]]).run_all()


def test_ranking_perfect_top1(tmp_path):
    relevant = [{1, 2}, {3}]
    ranked = [[1, 4, 5], [3, 1, 2]]
    report = RankingReport(relevant=relevant, ranked=ranked, k_values=(1, 2), output_dir=tmp_path)
    out = report.run_all()
    assert out["metrics"]["precision_at_1"] == 1.0
    assert out["metrics"]["hit_rate_at_1"] == 1.0
    assert out["metrics"]["map"] > 0.5
    assert "precision_at_k_curve" in report.plots
    assert Path(report.plots["precision_at_k_curve"]).exists()


def test_generate_report_ranking(tmp_path):
    out_path = tmp_path / "rec.html"
    generate_report(
        task="recommendation",
        y_true=[[10, 20], [30]],
        y_pred=[[10, 99, 20], [5, 30]],
        output_path=str(out_path),
        k_values=(1, 2),
    )
    assert out_path.exists()
    assert (tmp_path / "evalreport_plots").is_dir()
