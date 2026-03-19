from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..core.base_report import BaseReport


Box = Tuple[float, float, float, float]  # (x1, y1, x2, y2)


def _iou(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return 0.0 if union <= 0 else float(inter / union)


def _as_images_boxes(x: Optional[Iterable[Any]]) -> Optional[List[List[dict[str, Any]]]]:
    if x is None:
        return None
    # Expect list of images, each image list of dicts: {"bbox":[x1,y1,x2,y2], "label":..., "score":...}
    return [list(v) for v in x]


def _compute_ap_101point(precision: np.ndarray, recall: np.ndarray) -> float:
    """COCO-style AP using 101-point interpolation over recall in [0, 1]."""
    if precision.size == 0 or recall.size == 0:
        return 0.0
    # Ensure monotonic precision envelope
    p = precision.copy()
    for i in range(p.size - 2, -1, -1):
        p[i] = max(p[i], p[i + 1])
    rs = np.linspace(0.0, 1.0, 101)
    ap = 0.0
    for r in rs:
        mask = recall >= r
        ap += float(np.max(p[mask])) if np.any(mask) else 0.0
    return float(ap / 101.0)


def _coco_ap_for_class(
    gt: List[List[dict[str, Any]]],
    pr: List[List[dict[str, Any]]],
    *,
    label: Any,
    iou_threshold: float,
) -> tuple[Optional[float], np.ndarray, np.ndarray]:
    """Compute AP for a single class at a given IoU threshold.

    Returns (ap, precision_curve, recall_curve). If there are no GT boxes for
    this label, returns (None, empty, empty).
    """
    # Collect GT by image and count positives
    gt_by_img: List[List[Box]] = []
    npos = 0
    for boxes in gt:
        g = [tuple(b["bbox"]) for b in boxes if b.get("label", None) == label]
        gt_by_img.append(g)
        npos += len(g)
    if npos == 0:
        return None, np.array([]), np.array([])

    # Track matched GTs per image
    matched: List[List[bool]] = [[False] * len(g) for g in gt_by_img]

    # Collect predictions for this label across images
    preds: List[tuple[int, float, Box]] = []
    for img_idx, boxes in enumerate(pr):
        for b in boxes:
            if b.get("label", None) != label:
                continue
            score = float(b.get("score", 1.0))
            preds.append((img_idx, score, tuple(b["bbox"])))
    preds.sort(key=lambda t: t[1], reverse=True)

    tp = np.zeros(len(preds), dtype=float)
    fp = np.zeros(len(preds), dtype=float)

    for i, (img_idx, _score, pb) in enumerate(preds):
        gts = gt_by_img[img_idx]
        best_iou = 0.0
        best_j = -1
        for j, gb in enumerate(gts):
            if matched[img_idx][j]:
                continue
            iou = _iou(pb, gb)
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_j >= 0 and best_iou >= iou_threshold:
            matched[img_idx][best_j] = True
            tp[i] = 1.0
        else:
            fp[i] = 1.0

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-12)
    recall = cum_tp / float(npos)
    ap = _compute_ap_101point(precision, recall)
    return float(ap), precision, recall


@dataclass
class DetectionReport(BaseReport):
    y_true: Optional[Iterable[Any]] = None
    y_pred: Optional[Iterable[Any]] = None
    iou_threshold: float = 0.5
    max_visualizations: int = 3
    coco_iou_thresholds: Tuple[float, ...] = (
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
        0.95,
    )
    pr_curve_iou: float = 0.50

    def _compute_metrics(self) -> None:
        gt = _as_images_boxes(self.y_true)
        pr = _as_images_boxes(self.y_pred)
        if gt is None or pr is None:
            raise ValueError("DetectionReport requires y_true and y_pred (per-image box lists).")
        if len(gt) != len(pr):
            raise ValueError("DetectionReport requires y_true and y_pred with the same number of images.")

        tp = 0
        fp = 0
        fn = 0
        ious_matched: List[float] = []

        for gt_boxes, pr_boxes in zip(gt, pr):
            # group by label (optional); if missing, treat as single class
            # greedy matching by IoU
            gt_used = [False] * len(gt_boxes)

            # sort predictions by score desc if available
            pr_sorted = sorted(pr_boxes, key=lambda d: float(d.get("score", 1.0)), reverse=True)
            for pred in pr_sorted:
                pb = tuple(pred["bbox"])  # type: ignore[arg-type]
                pl = pred.get("label", None)
                best_iou = 0.0
                best_j = -1
                for j, g in enumerate(gt_boxes):
                    if gt_used[j]:
                        continue
                    if pl is not None and g.get("label", None) != pl:
                        continue
                    gb = tuple(g["bbox"])  # type: ignore[arg-type]
                    i = _iou(pb, gb)
                    if i > best_iou:
                        best_iou = i
                        best_j = j
                if best_j >= 0 and best_iou >= float(self.iou_threshold):
                    tp += 1
                    gt_used[best_j] = True
                    ious_matched.append(best_iou)
                else:
                    fp += 1

            fn += sum(1 for u in gt_used if not u)

        precision = 0.0 if (tp + fp) == 0 else tp / (tp + fp)
        recall = 0.0 if (tp + fn) == 0 else tp / (tp + fn)
        f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
        mean_iou = float(np.mean(ious_matched)) if ious_matched else 0.0

        # lightweight "AP" proxy: not COCO mAP; just F1/PR at a threshold.
        self.metrics.update(
            {
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "mean_matched_iou": float(mean_iou),
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "iou_threshold": float(self.iou_threshold),
            }
        )

        # COCO-style AP/mAP (label-aware; requires labels)
        labels = set()
        for boxes in gt:
            for b in boxes:
                labels.add(b.get("label", "__all__"))
        for boxes in pr:
            for b in boxes:
                labels.add(b.get("label", "__all__"))
        labels = {l if l is not None else "__all__" for l in labels}

        ap_by_thr: dict[str, float] = {}
        per_class_ap: dict[str, dict[str, float]] = {}
        for thr in self.coco_iou_thresholds:
            aps = []
            for lab in sorted(labels, key=lambda x: str(x)):
                ap, _p, _r = _coco_ap_for_class(gt, pr, label=lab, iou_threshold=float(thr))
                if ap is None:
                    continue
                aps.append(ap)
                per_class_ap.setdefault(str(lab), {})[f"ap@{thr:.2f}"] = float(ap)
            ap_by_thr[f"map@{thr:.2f}"] = float(np.mean(aps)) if aps else 0.0

        map_50_95 = float(np.mean(list(ap_by_thr.values()))) if ap_by_thr else 0.0
        map_50 = ap_by_thr.get("map@0.50", 0.0)
        map_75 = ap_by_thr.get("map@0.75", 0.0)

        self.metrics.update(
            {
                "map_50_95": map_50_95,
                "map_50": map_50,
                "map_75": map_75,
                "map_by_iou": ap_by_thr,
                "ap_by_class": per_class_ap,
            }
        )

        self.metric_descriptions.update(
            {
                "precision": "Fraction of predicted boxes that are correct matches (at IoU threshold).",
                "recall": "Fraction of ground-truth boxes recovered by predictions (at IoU threshold).",
                "f1": "Harmonic mean of precision and recall (single-threshold).",
                "mean_matched_iou": "Average IoU among matched true-positive boxes.",
                "tp": "True positives (matched predictions).",
                "fp": "False positives (unmatched predictions).",
                "fn": "False negatives (missed ground-truth boxes).",
                "iou_threshold": "IoU threshold used to determine a match.",
                "map_50_95": "COCO-style mAP averaged over IoU thresholds 0.50:0.95.",
                "map_50": "mAP at IoU=0.50.",
                "map_75": "mAP at IoU=0.75.",
                "map_by_iou": "mAP for each IoU threshold (0.50..0.95).",
                "ap_by_class": "Per-class AP values (by IoU threshold).",
            }
        )

        self._cached_gt = gt
        self._cached_pr = pr
        self._cached_labels = sorted(labels, key=lambda x: str(x))

    def _generate_plots(self) -> None:
        root = self.output_dir or Path("reports")
        plot_dir = root / "evalreport_plots"
        plot_dir.mkdir(parents=True, exist_ok=True)

        plots: dict[str, str] = {}
        gt = getattr(self, "_cached_gt", None)
        pr = getattr(self, "_cached_pr", None)
        if gt is None or pr is None:
            self.plots = plots
            return

        # Minimal visualization: draw GT vs Pred boxes on blank canvas per image index.
        n = min(int(self.max_visualizations), len(gt))
        for i in range(n):
            g = gt[i]
            p = pr[i]
            # determine canvas extents
            all_boxes = [b["bbox"] for b in g + p if "bbox" in b]
            if not all_boxes:
                continue
            xs = [c for bb in all_boxes for c in (bb[0], bb[2])]
            ys = [c for bb in all_boxes for c in (bb[1], bb[3])]
            w = max(xs) - min(xs)
            h = max(ys) - min(ys)
            w = max(1.0, w)
            h = max(1.0, h)

            plt.figure(figsize=(6, 6))
            ax = plt.gca()
            ax.set_xlim(min(xs) - 1, max(xs) + 1)
            ax.set_ylim(max(ys) + 1, min(ys) - 1)  # invert y
            ax.set_title("Detection: GT (green) vs Pred (red)")
            ax.set_xlabel("x")
            ax.set_ylabel("y")

            for bb in g:
                x1, y1, x2, y2 = bb["bbox"]
                rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color="green", linewidth=2)
                ax.add_patch(rect)
            for bb in p:
                x1, y1, x2, y2 = bb["bbox"]
                rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color="red", linewidth=2, linestyle="--")
                ax.add_patch(rect)

            path = plot_dir / f"detection_boxes_{i}.png"
            plt.tight_layout()
            plt.savefig(path)
            plt.close()
            plots[f"box_overlay_{i}"] = str(path)

        self.plots = plots

        # Precision-Recall curve for the first class at pr_curve_iou (if possible)
        try:
            labels = getattr(self, "_cached_labels", [])
            if labels:
                lab = labels[0]
                ap, precision, recall = _coco_ap_for_class(
                    gt,
                    pr,
                    label=lab,
                    iou_threshold=float(self.pr_curve_iou),
                )
                if ap is not None and precision.size > 0:
                    plt.figure(figsize=(5, 4))
                    plt.plot(recall, precision, label=f"PR (AP={ap:.3f})")
                    plt.xlabel("Recall")
                    plt.ylabel("Precision")
                    plt.title(f"PR Curve @ IoU={self.pr_curve_iou:.2f} (label={lab})")
                    plt.ylim(0, 1.05)
                    plt.xlim(0, 1.05)
                    plt.legend()
                    path = plot_dir / "detection_pr_curve.png"
                    plt.tight_layout()
                    plt.savefig(path)
                    plt.close()
                    self.plots["pr_curve"] = str(path)
        except Exception:
            pass

    def _generate_insights(self) -> None:
        insights: List[str] = []
        p = self.metrics.get("precision")
        r = self.metrics.get("recall")
        miou = self.metrics.get("mean_matched_iou")
        if isinstance(p, (int, float)) and isinstance(r, (int, float)):
            if p < 0.5 and r >= 0.7:
                insights.append("High recall but low precision: many false positives. Consider raising score threshold.")
            if r < 0.5 and p >= 0.7:
                insights.append("High precision but low recall: many missed objects. Consider lowering score threshold.")
        if isinstance(miou, (int, float)) and miou < 0.4:
            insights.append("Low mean IoU among matches suggests poor localization.")
        map_ = self.metrics.get("map_50_95")
        if isinstance(map_, (int, float)) and map_ < 0.2:
            insights.append("Low mAP suggests weak overall detection quality; verify label mapping and score thresholds.")
        self.insights = insights

