from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional

from ..classification.report import ClassificationReport
from ..clustering.report import ClusteringReport
from ..regression.report import RegressionReport
from ..timeseries.report import TimeSeriesReport
from .task_inference import infer_task


def generate_report(
    *,
    task: str = "auto",
    y_true: Optional[Iterable[Any]] = None,
    y_pred: Optional[Iterable[Any]] = None,
    y_prob: Optional[Iterable[Any]] = None,
    X: Optional[Iterable[Any]] = None,
    timestamps: Optional[Iterable[Any]] = None,
    embeddings: Optional[Iterable[Any]] = None,
    output_path: Optional[str] = None,
    format: str = "html",
    **kwargs: Any,
) -> Any:
    """Unified entry point for generating evaluation reports.

    This is a thin orchestrator that delegates to the appropriate
    task-specific report class based on the ``task`` argument or
    automatic task inference.
    """
    if task == "auto":
        task = infer_task(y_true=y_true, y_pred=y_pred, X=X, timestamps=timestamps, embeddings=embeddings)

    task = task.lower()

    # Determine base output directory for this report
    if output_path:
        base_dir = Path(output_path).expanduser().resolve().parent
    else:
        base_dir = Path("reports").resolve()
    base_dir.mkdir(parents=True, exist_ok=True)

    if task in {"classification", "binary_classification", "multiclass", "multilabel"}:
        report = ClassificationReport(y_true=y_true, y_pred=y_pred, y_prob=y_prob, **kwargs)
    elif task == "regression":
        report = RegressionReport(y_true=y_true, y_pred=y_pred, **kwargs)
    elif task in {"clustering", "cluster"}:
        # For clustering, y_pred is treated as cluster assignments/labels.
        report = ClusteringReport(X=X, labels=y_pred, **kwargs)
    elif task in {"timeseries", "forecasting", "time_series"}:
        report = TimeSeriesReport(y_true=y_true, y_pred=y_pred, timestamps=timestamps, **kwargs)
    else:
        raise ValueError(
            f"Unsupported task type for v0.1: {task!r}. "
            "Currently supported: classification, regression, clustering, timeseries."
        )

    # Make sure downstream plots know where they should live
    report.output_dir = base_dir

    result = report.run_all()

    # Persist report
    if output_path:
        output_path = str(output_path)
        suffix = Path(output_path).suffix.lower()
        fmt = format or suffix.lstrip(".") or "html"
        report.save(output_path, format=fmt)
    else:
        # Default to reports/<task>_report.html if user did not specify a path
        fmt = (format or "html").lower()
        default_name = f"{task}_report.{fmt}"
        default_path = base_dir / default_name
        report.save(str(default_path), format=fmt)

    return result

