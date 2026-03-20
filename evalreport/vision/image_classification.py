from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence

from ..classification.report import ClassificationReport


@dataclass
class ImageClassificationReport(ClassificationReport):
    """Image classification report.

    This is currently an alias of the core :class:`~evalreport.ClassificationReport`,
    because classification metrics/plots apply directly (confusion matrix,
    ROC/PR curves for binary with probabilities, etc.).

    Image-specific extensions (e.g., per-class error thumbnails) can be added
    in future versions without changing the public API.
    """

    y_true: Optional[Iterable[Any]] = None
    y_pred: Optional[Iterable[Any]] = None
    y_prob: Optional[Iterable[Any]] = None
    labels: Optional[Sequence[Any]] = None

