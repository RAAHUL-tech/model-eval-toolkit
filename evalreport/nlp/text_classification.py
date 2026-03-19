from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence

from ..classification.report import ClassificationReport


@dataclass
class TextClassificationReport(ClassificationReport):
    """Text classification report.

    Currently reuses the core ClassificationReport logic (metrics, plots, insights)
    and can be extended later with token-level error analysis.
    """

    y_true: Optional[Iterable[Any]] = None
    y_pred: Optional[Iterable[Any]] = None
    y_prob: Optional[Iterable[Any]] = None
    labels: Optional[Sequence[Any]] = None

