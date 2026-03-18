from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

        # MAPE (skip samples where |y_true| ~ 0; undefined if none left)
        valid = np.abs(y_true) >= 1e-12
        if valid.any():
            rel = np.abs((y_true[valid] - y_pred[valid]) / np.abs(y_true[valid]))
            self.metrics["mape"] = _safe_float(float(np.mean(rel) * 100.0))
        else:
            self.metrics["mape"] = None

        # Mean error (bias)
        self.metrics["mean_error"] = _safe_float(float(np.mean(y_pred - y_true)))

    def _generate_plots(self) -> None:
        y_true = _as_float_array(self.y_true)
        y_pred = _as_float_array(self.y_pred)
        if y_true is None or y_pred is None:
            self.plots = {}
            return

        root = self.output_dir or Path("reports")
        plot_dir = root / "evalreport_plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        plots: dict[str, str] = {}

        # Residual plot (residual vs predicted)
        try:
            residuals = y_pred - y_true
            plt.figure(figsize=(4, 3))
            plt.scatter(y_pred, residuals, alpha=0.7)
            plt.axhline(0, color="red", linestyle="--")
            plt.xlabel("Predicted")
            plt.ylabel("Residual (y_pred - y_true)")
            plt.title("Residual Plot")
            path = plot_dir / "regression_residuals.png"
            plt.tight_layout()
            plt.savefig(path)
            plt.close()
            plots["residual_plot"] = str(path)
        except Exception:
            pass

        # Predicted vs Actual
        try:
            plt.figure(figsize=(4, 3))
            plt.scatter(y_true, y_pred, alpha=0.7)
            min_v = float(min(np.min(y_true), np.min(y_pred)))
            max_v = float(max(np.max(y_true), np.max(y_pred)))
            plt.plot([min_v, max_v], [min_v, max_v], "k--", label="Ideal")
            plt.xlabel("Actual")
            plt.ylabel("Predicted")
            plt.title("Predicted vs Actual")
            plt.legend()
            path = plot_dir / "regression_pred_vs_actual.png"
            plt.tight_layout()
            plt.savefig(path)
            plt.close()
            plots["predicted_vs_actual"] = str(path)
        except Exception:
            pass

        # Error distribution
        try:
            residuals = y_pred - y_true
            plt.figure(figsize=(4, 3))
            plt.hist(residuals, bins=20, alpha=0.8)
            plt.xlabel("Residual")
            plt.ylabel("Count")
            plt.title("Error Distribution")
            path = plot_dir / "regression_error_distribution.png"
            plt.tight_layout()
            plt.savefig(path)
            plt.close()
            plots["error_distribution"] = str(path)
        except Exception:
            pass

        self.plots = plots

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

        # Descriptions for key metrics shown in HTML/PDF
        self.metric_descriptions.update(
            {
                "mae": "Mean absolute error; average absolute deviation between predictions and true values.",
                "mse": "Mean squared error; penalizes larger errors more strongly.",
                "rmse": "Root mean squared error; error in the same units as the target.",
                "r2": "Coefficient of determination; proportion of variance explained by the model.",
                "median_ae": "Median absolute error; robust to outliers in the error distribution.",
                "mape": "Mean absolute percentage error; average relative error as a percentage.",
                "mean_error": "Signed average error; indicates systematic over- or under-prediction.",
            }
        )

