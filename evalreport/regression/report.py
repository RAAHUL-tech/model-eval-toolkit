from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Optional

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)

from ..core.base_report import BaseReport


def _as_float_array(x: Optional[Iterable[Any]]) -> Optional[np.ndarray]:
    if x is None:
        return None
    arr = np.asarray(list(x), dtype=float)
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
class RegressionReport(BaseReport):
    y_true: Optional[Iterable[Any]] = None
    y_pred: Optional[Iterable[Any]] = None

    def _compute_metrics(self) -> None:
        y_true = _as_float_array(self.y_true)
        y_pred = _as_float_array(self.y_pred)
        if y_true is None or y_pred is None:
            raise ValueError("RegressionReport requires y_true and y_pred.")

        mse = mean_squared_error(y_true, y_pred)
        rmse = float(np.sqrt(mse))

        self.metrics["mae"] = _safe_float(mean_absolute_error(y_true, y_pred))
        self.metrics["mse"] = _safe_float(mse)
        self.metrics["rmse"] = _safe_float(rmse)
        self.metrics["r2"] = _safe_float(r2_score(y_true, y_pred))
        self.metrics["median_ae"] = _safe_float(median_absolute_error(y_true, y_pred))

        # MAPE (robust to zeros)
        denom = np.where(np.abs(y_true) < 1e-12, np.nan, np.abs(y_true))
        mape = np.nanmean(np.abs((y_true - y_pred) / denom)) * 100.0
        self.metrics["mape"] = _safe_float(mape) if not np.isnan(mape) else None

        # Mean error (bias)
        self.metrics["mean_error"] = _safe_float(float(np.mean(y_pred - y_true)))

    def _generate_plots(self) -> None:
        # Minimal v0.1: metrics-focused. Plotting hooks will be expanded later.
        self.plots = {}

    def _generate_insights(self) -> None:
        y_true = _as_float_array(self.y_true)
        y_pred = _as_float_array(self.y_pred)
        if y_true is None or y_pred is None:
            return

        insights: List[str] = []

        err = y_pred - y_true
        mean_err = float(np.mean(err))
        if abs(mean_err) > 0:
            direction = "positive" if mean_err > 0 else "negative"
            insights.append(f"Predictions show {direction} bias (mean error={mean_err:.4g}).")

        # Outlier influence heuristic: large 95th percentile absolute error
        abs_err = np.abs(err)
        p95 = float(np.percentile(abs_err, 95))
        med = float(np.median(abs_err))
        if med > 0 and (p95 / med) >= 5:
            insights.append(
                f"Errors have heavy tail (p95 abs error≈{p95:.4g} vs median≈{med:.4g}); check outliers."
            )

        self.insights = insights

