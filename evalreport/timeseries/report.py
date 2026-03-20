from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..core.base_report import BaseReport


def _as_float_1d(x: Optional[Iterable[Any]]) -> Optional[np.ndarray]:
    if x is None:
        return None
    arr = np.asarray(list(x), dtype=float).reshape(-1)
    return arr


def _as_1d(x: Optional[Iterable[Any]]) -> Optional[np.ndarray]:
    if x is None:
        return None
    return np.asarray(list(x))


@dataclass
class TimeSeriesReport(BaseReport):
    y_true: Optional[Iterable[Any]] = None
    y_pred: Optional[Iterable[Any]] = None
    timestamps: Optional[Iterable[Any]] = None
    rolling_window: int = 5

    def _compute_metrics(self) -> None:
        y_true = _as_float_1d(self.y_true)
        y_pred = _as_float_1d(self.y_pred)
        ts = _as_1d(self.timestamps)
        if y_true is None or y_pred is None or ts is None:
            raise ValueError("TimeSeriesReport requires y_true, y_pred, and timestamps.")
        if not (len(y_true) == len(y_pred) == len(ts)):
            raise ValueError("TimeSeriesReport requires y_true, y_pred, timestamps with equal length.")
        if len(y_true) < 2:
            raise ValueError("TimeSeriesReport requires at least 2 samples.")

        err = y_pred - y_true
        abs_err = np.abs(err)

        mae = float(np.mean(abs_err))
        mse = float(np.mean(err**2))
        rmse = float(np.sqrt(mse))

        denom = np.where(np.abs(y_true) < 1e-12, np.nan, np.abs(y_true))
        mape = float(np.nanmean(abs_err / denom) * 100.0)
        if np.isnan(mape):
            mape = None  # undefined when all targets are ~0

        smape_denom = np.abs(y_true) + np.abs(y_pred)
        smape = float(np.mean(np.where(smape_denom < 1e-12, np.nan, 2.0 * abs_err / smape_denom)) * 100.0)
        if np.isnan(smape):
            smape = None

        mean_forecast_error = float(np.mean(err))

        # Rolling RMSE
        w = int(self.rolling_window)
        w = max(2, min(w, len(y_true)))
        rolling = []
        for i in range(w - 1, len(y_true)):
            window_err = err[i - w + 1 : i + 1]
            rolling.append(float(np.sqrt(np.mean(window_err**2))))

        # Cache the rolling window actually used so plots align with metrics.
        self._cached_rolling_window = w

        rolling_rmse_last = float(rolling[-1]) if rolling else None
        rolling_rmse_mean = float(np.mean(rolling)) if rolling else None

        self.metrics.update(
            {
                "mae": mae,
                "mse": mse,
                "rmse": rmse,
                "mape": mape,
                "smape": smape,
                "mean_forecast_error": mean_forecast_error,
                "rolling_rmse_mean": rolling_rmse_mean,
                "rolling_rmse_last": rolling_rmse_last,
            }
        )


        # Store for plotting
        self._cached_ts = ts
        self._cached_y_true = y_true
        self._cached_y_pred = y_pred
        self._cached_rolling_rmse = rolling

    def _generate_plots(self) -> None:
        y_true = _as_float_1d(self.y_true)
        y_pred = _as_float_1d(self.y_pred)
        ts = _as_1d(self.timestamps)
        if y_true is None or y_pred is None or ts is None:
            self.plots = {}
            return
        if len(y_true) != len(y_pred) or len(y_true) != len(ts):
            self.plots = {}
            return

        root = self.output_dir or Path("reports")
        plot_dir = root / "evalreport_plots"
        plot_dir.mkdir(parents=True, exist_ok=True)

        # Ensure chronological order
        order = np.argsort(ts)
        ts_sorted = ts[order]
        y_true_sorted = y_true[order]
        y_pred_sorted = y_pred[order]
        residuals = y_pred_sorted - y_true_sorted

        # Rolling RMSE aligns with timestamps after (w-1) offset.
        rolling = getattr(self, "_cached_rolling_rmse", None)
        w = getattr(self, "_cached_rolling_window", max(2, int(self.rolling_window)))
        if len(ts_sorted) >= w:
            rolling_ts = ts_sorted[w - 1 :]
        else:
            # Extremely short series; align lengths defensively.
            rolling_ts = ts_sorted[: len(rolling)] if rolling is not None else ts_sorted

        plots: dict[str, str] = {}

        # Actual vs Forecast
        plt.figure(figsize=(6, 4))
        plt.plot(ts_sorted, y_true_sorted, label="Actual", linewidth=2)
        plt.plot(ts_sorted, y_pred_sorted, label="Forecast", linewidth=2, linestyle="--")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.title("Actual vs Forecast")
        plt.legend()
        path = plot_dir / "timeseries_actual_vs_forecast.png"
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        plots["actual_vs_forecast"] = str(path)

        # Residuals over time
        plt.figure(figsize=(6, 3.8))
        plt.plot(ts_sorted, residuals, label="Residual", color="#D55E00")
        plt.axhline(0, color="black", linewidth=1)
        plt.xlabel("Time")
        plt.ylabel("Residual (y_pred - y_true)")
        plt.title("Residuals over time")
        plt.legend()
        path = plot_dir / "timeseries_residuals_over_time.png"
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        plots["residuals_over_time"] = str(path)

        # Rolling RMSE
        if rolling is not None and len(rolling) > 0:
            plt.figure(figsize=(6, 3.8))
            plt.plot(rolling_ts, rolling, label="Rolling RMSE", color="#0072B2")
            plt.xlabel("Time")
            plt.ylabel("RMSE")
            plt.title(f"Rolling RMSE (window={self.rolling_window})")
            plt.legend()
            path = plot_dir / "timeseries_rolling_rmse.png"
            plt.tight_layout()
            plt.savefig(path)
            plt.close()
            plots["rolling_rmse"] = str(path)

        self.plots = plots

    def _generate_insights(self) -> None:
        y_true = _as_float_1d(self.y_true)
        y_pred = _as_float_1d(self.y_pred)
        if y_true is None or y_pred is None:
            self.insights = []
            return
        err = y_pred - y_true
        mean_err = float(np.mean(err))
        insights: List[str] = []
        if abs(mean_err) > 0:
            direction = "over" if mean_err > 0 else "under"
            insights.append(f"Forecasts show systematic {direction}-prediction bias (mean error={mean_err:.4g}).")

        rolling_mean = self.metrics.get("rolling_rmse_mean")
        rolling_last = self.metrics.get("rolling_rmse_last")
        if isinstance(rolling_mean, (int, float)) and isinstance(rolling_last, (int, float)) and rolling_mean:
            if rolling_last > 1.2 * rolling_mean:
                insights.append("Recent errors increased vs earlier windows; model may be drifting.")
            elif rolling_last < 0.8 * rolling_mean:
                insights.append("Recent errors improved vs earlier windows.")

        self.insights = insights

        self.metric_descriptions.update(
            {
                "mae": "Mean absolute error; average absolute deviation between forecast and truth.",
                "rmse": "Root mean squared error; penalizes larger errors more.",
                "mape": "Mean absolute percentage error; undefined if true values are all ~0.",
                "smape": "Symmetric MAPE; more stable when scaling varies between truth and prediction.",
                "mean_forecast_error": "Signed average error; indicates systematic over- or under-forecasting.",
                "rolling_rmse_mean": "Average RMSE across rolling windows (overall stability).",
                "rolling_rmse_last": "Most recent rolling RMSE (latest stability).",
            }
        )

