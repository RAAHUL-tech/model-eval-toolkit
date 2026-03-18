from .core.entrypoints import generate_report
from .classification.report import ClassificationReport
from .regression.report import RegressionReport
from .__version__ import __version__

__all__ = [
    "generate_report",
    "__version__",
    "ClassificationReport",
    "RegressionReport",
]

