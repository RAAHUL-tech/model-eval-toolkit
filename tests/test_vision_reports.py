from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from evalreport import DetectionReport, SegmentationReport


def test_segmentation_metrics_and_plots(tmp_path):
    # 2 samples, 8x8 masks
    yt = np.zeros((2, 8, 8), dtype=np.uint8)
    yp = np.zeros((2, 8, 8), dtype=np.uint8)
    yt[0, 2:6, 2:6] = 1
    yp[0, 3:7, 3:7] = 1  # shifted overlap
    yt[1, 1:3, 1:3] = 1
    yp[1, 1:3, 1:3] = 1

    report = SegmentationReport(y_true_masks=yt, y_pred_masks=yp, max_visualizations=2)
    report.output_dir = tmp_path
    out = report.run_all()
    assert 0.0 <= out["metrics"]["mean_iou"] <= 1.0
    assert 0.0 <= out["metrics"]["mean_dice"] <= 1.0
    assert "mask_comparison_0" in report.plots
    assert Path(report.plots["mask_comparison_0"]).exists()


def test_segmentation_shape_mismatch_raises():
    yt = np.zeros((1, 4, 4), dtype=np.uint8)
    yp = np.zeros((2, 4, 4), dtype=np.uint8)
    with pytest.raises(ValueError, match="same shape"):
        SegmentationReport(y_true_masks=yt, y_pred_masks=yp).run_all()


def test_detection_metrics_and_overlay(tmp_path):
    # One image with 1 GT box and 2 predicted boxes (one match, one FP)
    y_true = [
        [
            {"bbox": [0, 0, 10, 10], "label": "obj"},
        ]
    ]
    y_pred = [
        [
            {"bbox": [1, 1, 9, 9], "label": "obj", "score": 0.9},  # TP
            {"bbox": [20, 20, 30, 30], "label": "obj", "score": 0.2},  # FP
        ]
    ]

    report = DetectionReport(y_true=y_true, y_pred=y_pred, iou_threshold=0.5, max_visualizations=1)
    report.output_dir = tmp_path
    out = report.run_all()
    assert out["metrics"]["tp"] == 1
    assert out["metrics"]["fp"] == 1
    assert out["metrics"]["fn"] == 0
    assert "box_overlay_0" in report.plots
    assert Path(report.plots["box_overlay_0"]).exists()


def test_detection_requires_same_num_images():
    with pytest.raises(ValueError, match="same number of images"):
        DetectionReport(y_true=[[]], y_pred=[[], []]).run_all()

