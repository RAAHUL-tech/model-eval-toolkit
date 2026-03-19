from __future__ import annotations

import pytest

from evalreport import DetectionReport


def test_detection_map_perfect_is_one(tmp_path):
    # Two images, one GT each, perfect predictions with high score
    y_true = [
        [{"bbox": [0, 0, 10, 10], "label": "obj"}],
        [{"bbox": [5, 5, 15, 15], "label": "obj"}],
    ]
    y_pred = [
        [{"bbox": [0, 0, 10, 10], "label": "obj", "score": 0.9}],
        [{"bbox": [5, 5, 15, 15], "label": "obj", "score": 0.8}],
    ]
    r = DetectionReport(y_true=y_true, y_pred=y_pred, max_visualizations=0)
    r.output_dir = tmp_path
    out = r.run_all()
    assert out["metrics"]["map_50_95"] == pytest.approx(1.0, rel=1e-9)
    assert out["metrics"]["map_50"] == pytest.approx(1.0, rel=1e-9)
    assert out["metrics"]["map_75"] == pytest.approx(1.0, rel=1e-9)


def test_detection_map_iou_sensitive(tmp_path):
    # IoU around ~0.68: should be TP at 0.50 but fail at 0.75
    y_true = [[{"bbox": [0, 0, 10, 10], "label": "obj"}]]
    # IoU = 64/100 = 0.64
    y_pred = [[{"bbox": [2, 2, 10, 10], "label": "obj", "score": 0.9}]]
    r = DetectionReport(y_true=y_true, y_pred=y_pred, max_visualizations=0)
    r.output_dir = tmp_path
    out = r.run_all()
    assert out["metrics"]["map_50"] == pytest.approx(1.0, rel=1e-9)
    assert out["metrics"]["map_75"] == pytest.approx(0.0, rel=1e-9)


def test_detection_map_multiclass_average(tmp_path):
    y_true = [
        [{"bbox": [0, 0, 10, 10], "label": "a"}, {"bbox": [20, 20, 30, 30], "label": "b"}]
    ]
    y_pred = [
        [
            {"bbox": [0, 0, 10, 10], "label": "a", "score": 0.9},  # correct
            {"bbox": [20, 20, 30, 30], "label": "b", "score": 0.8},  # correct
        ]
    ]
    r = DetectionReport(y_true=y_true, y_pred=y_pred, max_visualizations=0)
    r.output_dir = tmp_path
    out = r.run_all()
    assert out["metrics"]["map_50_95"] == pytest.approx(1.0, rel=1e-9)
    assert "ap_by_class" in out["metrics"]
    assert "a" in out["metrics"]["ap_by_class"]
    assert "b" in out["metrics"]["ap_by_class"]

