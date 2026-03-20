from .core.entrypoints import generate_report
from .classification.report import ClassificationReport
from .regression.report import RegressionReport
from .__version__ import __version__
from .clustering.report import ClusteringReport
from .timeseries.report import TimeSeriesReport
from .nlp.text_classification import TextClassificationReport
from .nlp.text_generation import TextGenerationReport
from .vision.segmentation import SegmentationReport
from .vision.detection import DetectionReport
from .vision.image_classification import ImageClassificationReport
from .ranking.report import RankingReport

__all__ = [
    "generate_report",
    "__version__",
    "ClassificationReport",
    "RegressionReport",
    "ClusteringReport",
    "TimeSeriesReport",
    "TextClassificationReport",
    "TextGenerationReport",
    "SegmentationReport",
    "DetectionReport",
    "ImageClassificationReport",
    "RankingReport",
]

