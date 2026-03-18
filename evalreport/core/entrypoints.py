from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional
from ..classification.report import ClassificationReport
from ..regression.report import RegressionReport
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

    if task in {"classification", "binary_classification", "multiclass", "multilabel"}:
        report = ClassificationReport(y_true=y_true, y_pred=y_pred, y_prob=y_prob, **kwargs)
    elif task == "regression":
        report = RegressionReport(y_true=y_true, y_pred=y_pred, **kwargs)
    else:
        raise ValueError(
            f"Unsupported task type for v0.1: {task!r}. "
            "Currently supported: classification, regression."
        )

    result = report.run_all()

    if output_path:
        output_path = str(output_path)
        suffix = Path(output_path).suffix.lower()
        fmt = format or suffix.lstrip(".") or "html"
        report.save(output_path, format=fmt)

    return result

