from __future__ import annotations
from typing import Any, Iterable, Optional

import numpy as np


def infer_task(
    *,
    y_true: Optional[Iterable[Any]] = None,
    y_pred: Optional[Iterable[Any]] = None,
    X: Optional[Iterable[Any]] = None,
    timestamps: Optional[Iterable[Any]] = None,
    embeddings: Optional[Iterable[Any]] = None,
    y_prob: Optional[Iterable[Any]] = None,
) -> str:
    """Infer ML task type from provided inputs.

    Notes
    -----
    This is a heuristic v0.1 inference intended to be *good enough* for common
    scenarios. Power users should still prefer `task="classification"` /
    `task="regression"` / etc. when they know the correct task.
    """
    if timestamps is not None:
        return "timeseries"
    if embeddings is not None:
        return "clustering"

    def _is_seq_of_strings(x: Any) -> bool:
        if x is None:
            return False
        try:
            seq = list(x)
            if not seq:
                return False
            return all(isinstance(v, str) for v in seq)
        except Exception:
            return False

    def _avg_tokens(seq: Iterable[Any]) -> float:
        vals = list(seq)
        if not vals:
            return 0.0
        total = 0
        n = 0
        for v in vals:
            if v is None:
                continue
            s = str(v)
            total += len(s.split())
            n += 1
        return (total / max(1, n)) if n else 0.0

    def _looks_like_text_generation(y_t: Any, y_p: Any) -> bool:
        # Both sides are strings; generation outputs tend to be longer.
        if not (_is_seq_of_strings(y_t) and _is_seq_of_strings(y_p)):
            return False
        avg_t = _avg_tokens(y_t)
        avg_p = _avg_tokens(y_p)
        # Conservative threshold: labels like "pos"/"neg" usually have ~1 token.
        return (avg_t > 3.0) or (avg_p > 3.0)

    def _looks_like_text_classification(y_t: Any, y_p: Any) -> bool:
        if not (_is_seq_of_strings(y_t) and _is_seq_of_strings(y_p)):
            return False
        # If they are strings but short, treat as classification labels.
        avg_t = _avg_tokens(y_t)
        avg_p = _avg_tokens(y_p)
        return (avg_t <= 3.0) and (avg_p <= 3.0)

    def _looks_like_detection(y_t: Any, y_p: Any) -> bool:
        # Expect: list of images -> list of dict boxes with 'bbox'
        try:
            gt = list(y_t) if y_t is not None else None
            pr = list(y_p) if y_p is not None else None
            if not gt or not pr:
                return False
            if not isinstance(gt[0], (list, tuple)) or not isinstance(pr[0], (list, tuple)):
                return False
            # Check first non-empty box dict
            for boxes in (gt[0], pr[0]):
                if not boxes:
                    continue
                first = boxes[0]
                if isinstance(first, dict) and "bbox" in first:
                    return True
            return False
        except Exception:
            return False

    def _looks_like_segmentation(y_t: Any, y_p: Any) -> bool:
        # Expect: masks with at least 2D structure (H,W) or (N,H,W)
        try:
            gt = np.asarray(list(y_t)) if y_t is not None else None
            pr = np.asarray(list(y_p)) if y_p is not None else None
            if gt is None or pr is None:
                return False
            # classification labels are 1D; segmentation should be 2D or 3D
            return gt.ndim >= 2 and pr.ndim >= 2
        except Exception:
            return False

    def _looks_like_recommendation(y_t: Any, y_p: Any) -> bool:
        # Expect per-user relevant items + ranked lists:
        # y_true: list[ list[item] ] (or set)
        # y_pred: list[ list[item] ]
        try:
            gt = list(y_t) if y_t is not None else None
            pr = list(y_p) if y_p is not None else None
            if not gt or not pr:
                return False
            if not isinstance(gt[0], (list, tuple, set)):
                return False
            if not isinstance(pr[0], (list, tuple)):
                return False
            return True
        except Exception:
            return False

    def _looks_like_discrete_labels(arr: np.ndarray) -> bool:
        # If labels are integer-like (or small number of unique values),
        # likely classification/clustering.
        if arr.size == 0:
            return False
        if arr.dtype.kind in {"i", "u", "b"}:
            return True
        if arr.dtype.kind == "f":
            # float but close to integers?
            rounded = np.rint(arr)
            if np.allclose(arr, rounded, atol=1e-8):
                return True
        # otherwise: not enough evidence
        return False

    # Vision --------------------------------------------------------------
    if y_true is not None and y_pred is not None:
        if _looks_like_detection(y_true, y_pred):
            return "detection"
        if _looks_like_segmentation(y_true, y_pred):
            return "segmentation"
        if _looks_like_recommendation(y_true, y_pred):
            return "recommendation"

        # NLP --------------------------------------------------------------
        if _looks_like_text_generation(y_true, y_pred):
            return "text_generation"
        if _looks_like_text_classification(y_true, y_pred):
            return "text_classification"

    # Clustering ----------------------------------------------------------
    if X is not None and y_true is None and y_pred is not None:
        try:
            labels = np.asarray(list(y_pred))
            if _looks_like_discrete_labels(labels) and labels.ndim == 1:
                return "clustering"
        except Exception:
            pass

    # Regression vs classification ---------------------------------------
    if y_true is not None and y_pred is not None:
        try:
            y_true_arr = np.asarray(list(y_true))
            if y_true_arr.ndim == 0:
                return "classification"

            if y_true_arr.dtype.kind == "f":
                # If float targets look integer-like, treat as classification labels
                # only when predictions are also integer-like (otherwise it's regression).
                y_pred_arr = np.asarray(list(y_pred))
                if _looks_like_discrete_labels(y_true_arr) and _looks_like_discrete_labels(y_pred_arr):
                    return "classification"
                return "regression"
        except Exception:  # pragma: no cover - heuristic fallback
            pass

        # Non-float targets default to classification.
        return "classification"

    return "classification"

