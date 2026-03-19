from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..core.base_report import BaseReport


def _as_mask_array(x: Optional[Iterable[Any]]) -> Optional[np.ndarray]:
    if x is None:
        return None
    arr = np.asarray(list(x))
    # Accept (H,W) single mask or (N,H,W)
    if arr.ndim == 2:
        arr = arr[None, ...]
    return arr


def _binarize(mask: np.ndarray) -> np.ndarray:
    # if mask already 0/1, keep; else threshold at 0.5 for float masks
    if mask.dtype.kind in {"f"}:
        return (mask >= 0.5).astype(np.uint8)
    return (mask > 0).astype(np.uint8)


@dataclass
class SegmentationReport(BaseReport):
    y_true_masks: Optional[Iterable[Any]] = None
    y_pred_masks: Optional[Iterable[Any]] = None
    max_visualizations: int = 3

    def _compute_metrics(self) -> None:
        yt = _as_mask_array(self.y_true_masks)
        yp = _as_mask_array(self.y_pred_masks)
        if yt is None or yp is None:
            raise ValueError("SegmentationReport requires y_true_masks and y_pred_masks.")
        if yt.shape != yp.shape:
            raise ValueError("SegmentationReport requires y_true_masks and y_pred_masks with the same shape.")

        yt_b = _binarize(yt)
        yp_b = _binarize(yp)

        # Flatten per-sample
        yt_f = yt_b.reshape(yt_b.shape[0], -1)
        yp_f = yp_b.reshape(yp_b.shape[0], -1)

        ious = []
        dices = []
        pix_accs = []
        for t, p in zip(yt_f, yp_f):
            inter = int(np.sum((t == 1) & (p == 1)))
            union = int(np.sum((t == 1) | (p == 1)))
            iou = 1.0 if union == 0 else inter / union
            denom = int(np.sum(t) + np.sum(p))
            dice = 1.0 if denom == 0 else (2 * inter) / denom
            pix_acc = float(np.mean(t == p))
            ious.append(iou)
            dices.append(dice)
            pix_accs.append(pix_acc)

        self.metrics.update(
            {
                "mean_iou": float(np.mean(ious)) if ious else 0.0,
                "mean_dice": float(np.mean(dices)) if dices else 0.0,
                "mean_pixel_accuracy": float(np.mean(pix_accs)) if pix_accs else 0.0,
                "num_samples": int(yt.shape[0]),
            }
        )

        self.metric_descriptions.update(
            {
                "mean_iou": "Intersection over Union (Jaccard) averaged over samples; higher is better.",
                "mean_dice": "Dice coefficient averaged over samples; higher is better.",
                "mean_pixel_accuracy": "Fraction of pixels predicted correctly (foreground/background).",
                "num_samples": "Number of masks evaluated.",
            }
        )

        self._cached_masks = (yt_b, yp_b)

    def _generate_plots(self) -> None:
        root = self.output_dir or Path("reports")
        plot_dir = root / "evalreport_plots"
        plot_dir.mkdir(parents=True, exist_ok=True)

        plots: dict[str, str] = {}
        yt_b, yp_b = getattr(self, "_cached_masks", (None, None))
        if yt_b is None or yp_b is None:
            self.plots = plots
            return

        n = min(int(self.max_visualizations), yt_b.shape[0])
        for i in range(n):
            t = yt_b[i]
            p = yp_b[i]
            overlay = np.zeros((*t.shape, 3), dtype=np.uint8)
            # true in green, pred in red, overlap yellow
            overlay[..., 1] = (t * 180).astype(np.uint8)
            overlay[..., 0] = (p * 180).astype(np.uint8)
            overlay[..., 2] = ((t & p) * 50).astype(np.uint8)

            plt.figure(figsize=(9, 3))
            plt.subplot(1, 3, 1)
            plt.imshow(t, cmap="gray")
            plt.title("Ground truth")
            plt.axis("off")
            plt.subplot(1, 3, 2)
            plt.imshow(p, cmap="gray")
            plt.title("Prediction")
            plt.axis("off")
            plt.subplot(1, 3, 3)
            plt.imshow(overlay)
            plt.title("Overlay")
            plt.axis("off")
            path = plot_dir / f"segmentation_masks_{i}.png"
            plt.tight_layout()
            plt.savefig(path)
            plt.close()
            plots[f"mask_comparison_{i}"] = str(path)

        self.plots = plots

    def _generate_insights(self) -> None:
        insights: List[str] = []
        iou = self.metrics.get("mean_iou")
        if isinstance(iou, (int, float)):
            if iou < 0.3:
                insights.append("Low IoU suggests poor localization/shape overlap; inspect mask alignment and thresholds.")
            elif iou > 0.7:
                insights.append("High IoU indicates strong mask overlap and good segmentation quality.")
        self.insights = insights

