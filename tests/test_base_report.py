"""BaseReport save / format behavior."""

from __future__ import annotations

import pytest

from evalreport import ClassificationReport


def test_unsupported_save_format_raises(tmp_path):
    report = ClassificationReport(y_true=[0, 1], y_pred=[0, 1])
    report.run_all()
    with pytest.raises(ValueError, match="Unsupported output format"):
        report.save(tmp_path / "out.xyz", format="xyz")


def test_pdf_save_creates_file(tmp_path):
    pytest.importorskip("reportlab")
    report = ClassificationReport(y_true=[0, 1, 0, 1], y_pred=[0, 1, 1, 0])
    report.run_all()
    pdf = tmp_path / "r.pdf"
    report.save(pdf, format="pdf")
    assert pdf.exists()
    assert pdf.stat().st_size > 100
