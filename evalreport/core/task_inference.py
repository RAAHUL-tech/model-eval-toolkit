from __future__ import annotations

from typing import Any, Iterable, Optional


def infer_task(
    *,
    y_true: Optional[Iterable[Any]] = None,
    y_pred: Optional[Iterable[Any]] = None,
    X: Optional[Iterable[Any]] = None,
    timestamps: Optional[Iterable[Any]] = None,
    embeddings: Optional[Iterable[Any]] = None,
) -> str:
    """Very lightweight heuristic task inference.

    The goal for v0.1 is not perfect detection but a reasonable default
    that power users can override via ``task=...``.
    """
    if timestamps is not None:
        return "timeseries"
    if embeddings is not None:
        return "clustering"
    if y_true is not None and y_pred is not None:
        # Simple heuristic: if y_true looks continuous -> regression
        try:
            import numpy as np

            y_true_arr = np.asarray(list(y_true))
            if y_true_arr.dtype.kind in {"f"}:
                return "regression"
        except Exception:  # pragma: no cover - heuristic fallback
            pass
        return "classification"
    return "classification"

