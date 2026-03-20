# API overview

## Public package surface

```python
from evalreport import (
    generate_report,
    __version__,
    ClassificationReport,
    RegressionReport,
    ClusteringReport,
    TimeSeriesReport,
    TextClassificationReport,
    TextGenerationReport,
    SegmentationReport,
    DetectionReport,
    ImageClassificationReport,
    RankingReport,
)
```

## `generate_report`

High-level orchestrator: infers or selects a task, sets `output_dir` from `output_path`, runs the report, saves, returns `to_dict()` output.  
→ [generate-report.md](generate-report.md)

## Task-specific reports

Each report is a subclass of **`BaseReport`** (`evalreport.core.base_report`):

| Method | Purpose |
|--------|---------|
| `run_all()` | Compute metrics, generate plots, build insights; returns summary dict. |
| `to_dict()` | JSON-serializable summary including `metric_descriptions`. |
| `save(path, format=...)` | Write HTML, JSON, Markdown, or PDF. |
| `output_dir` | Set before `run_all()` if you need plots under a specific folder. |

### Typical direct usage

```python
from pathlib import Path
from evalreport import ClassificationReport

r = ClassificationReport(y_true=[0, 1], y_pred=[0, 0], y_prob=[0.8, 0.3])
r.output_dir = Path("reports")
r.run_all()
r.save("reports/classification_report.html", format="html")
```

## Task inference

`evalreport.core.task_inference.infer_task` — used when `task="auto"`.  
→ [task-inference.md](task-inference.md)

## Version

```python
from evalreport import __version__
```

Also defined in `evalreport/__version__.py` and should stay aligned with `pyproject.toml` for releases.
