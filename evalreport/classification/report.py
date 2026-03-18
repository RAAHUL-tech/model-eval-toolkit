from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)

from ..core.base_report import BaseReport


def _as_array(x: Optional[Iterable[Any]]) -> Optional[np.ndarray]:
    if x is None:
        return None
    arr = np.asarray(list(x))
    return arr


def _safe_float(x: Any) -> Any:
    try:
        if isinstance(x, (np.floating, np.integer)):
            return x.item()
        if isinstance(x, float) and (np.isnan(x) or np.isinf(x)):
            return None
        return float(x)
    except Exception:
        return x


@dataclass
class ClassificationReport(BaseReport):
    y_true: Optional[Iterable[Any]] = None
    y_pred: Optional[Iterable[Any]] = None
    y_prob: Optional[Iterable[Any]] = None
    labels: Optional[Sequence[Any]] = None

    def _compute_metrics(self) -> None:
        y_true = _as_array(self.y_true)
        y_pred = _as_array(self.y_pred)
        if y_true is None or y_pred is None:
            raise ValueError("ClassificationReport requires y_true and y_pred.")

        average_modes = ["micro", "macro", "weighted"]
        self.metrics["accuracy"] = _safe_float(accuracy_score(y_true, y_pred))

        for avg in average_modes:
            self.metrics[f"precision_{avg}"] = _safe_float(
                precision_score(y_true, y_pred, average=avg, zero_division=0)
            )
            self.metrics[f"recall_{avg}"] = _safe_float(
                recall_score(y_true, y_pred, average=avg, zero_division=0)
            )
            self.metrics[f"f1_{avg}"] = _safe_float(
                f1_score(y_true, y_pred, average=avg, zero_division=0)
            )

        # Extras
        self.metrics["mcc"] = _safe_float(matthews_corrcoef(y_true, y_pred))
        try:
            self.metrics["cohen_kappa"] = _safe_float(cohen_kappa_score(y_true, y_pred))
        except Exception:
            self.metrics["cohen_kappa"] = None

        # Probabilistic metrics (best-effort; may be None)
        y_prob = _as_array(self.y_prob)
        if y_prob is not None:
            try:
                self.metrics["log_loss"] = _safe_float(log_loss(y_true, y_prob, labels=self.labels))
            except Exception:
                self.metrics["log_loss"] = None

            # ROC/PR AUC (binary or multiclass if possible)
            try:
                # Binary case: accept shape (n,) or (n,2) and use positive class scores
                if y_prob.ndim == 1:
                    y_score = y_prob
                elif y_prob.ndim == 2 and y_prob.shape[1] == 2:
                    y_score = y_prob[:, 1]
                else:
                    y_score = y_prob

                self.metrics["roc_auc"] = _safe_float(
                    roc_auc_score(y_true, y_score, multi_class="ovr" if getattr(y_score, "ndim", 1) == 2 else "raise")
                )
            except Exception:
                self.metrics["roc_auc"] = None

            try:
                if y_prob.ndim == 1:
                    y_score = y_prob
                elif y_prob.ndim == 2 and y_prob.shape[1] == 2:
                    y_score = y_prob[:, 1]
                else:
                    y_score = y_prob

                # average_precision_score supports binary directly; multiclass handled as macro if possible
                self.metrics["pr_auc"] = _safe_float(
                    average_precision_score(
                        y_true,
                        y_score,
                        average="macro" if getattr(y_score, "ndim", 1) == 2 else "macro",
                    )
                )
            except Exception:
                self.metrics["pr_auc"] = None

        # Confusion matrix as a small, JSON-serializable payload
        try:
            cm = confusion_matrix(y_true, y_pred, labels=self.labels)
            self.metrics["confusion_matrix"] = cm.tolist()
        except Exception:
            self.metrics["confusion_matrix"] = None

    def _generate_plots(self) -> None:
        # Minimal v0.1: metrics-focused. Plotting hooks will be expanded later.
        self.plots = {}

    def _generate_insights(self) -> None:
        y_true = _as_array(self.y_true)
        y_pred = _as_array(self.y_pred)
        if y_true is None or y_pred is None:
            return

        insights: List[str] = []

        # Class imbalance detection (simple heuristic)
        try:
            values, counts = np.unique(y_true, return_counts=True)
            if len(counts) > 1:
                ratio = counts.max() / max(1, counts.min())
                if ratio >= 5:
                    minority = values[np.argmin(counts)]
                    majority = values[np.argmax(counts)]
                    insights.append(
                        f"Class imbalance detected (majority={majority!r}, minority={minority!r}, ratio≈{ratio:.1f})."
                    )
        except Exception:
            pass

        # Misclassification trends: top confusions off-diagonal
        try:
            labels = self.labels
            if labels is None:
                labels = list(np.unique(np.concatenate([y_true, y_pred])))
            cm = confusion_matrix(y_true, y_pred, labels=labels)
            cm_off = cm.copy()
            np.fill_diagonal(cm_off, 0)
            if cm_off.sum() > 0:
                i, j = np.unravel_index(np.argmax(cm_off), cm_off.shape)
                if cm_off[i, j] > 0:
                    insights.append(
                        f"Most common confusion: true={labels[i]!r} predicted={labels[j]!r} ({int(cm_off[i, j])} samples)."
                    )
        except Exception:
            pass

        self.insights = insights

