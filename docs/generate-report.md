# `generate_report` reference

Unified entry point: picks (or infers) a task, builds the right report, saves to disk, and returns a summary dict.

```python
from evalreport import generate_report

generate_report(
    task="auto",                    # or explicit task name — see tasks.md
    y_true=...,                   # ground truth (task-dependent)
    y_pred=...,                   # predictions (task-dependent)
    y_prob=None,                  # optional class probabilities
    X=None,                       # features (clustering)
    timestamps=None,              # time series
    embeddings=None,              # alternative clustering hint (inference)
    output_path=None,             # if None → reports/<task>_report.<fmt>
    format="html",                # html | json | markdown | pdf (or infer from path suffix)
    **kwargs,                     # passed to the task-specific report class
)
```

## Parameters

| Parameter | Description |
|-----------|-------------|
| `task` | Task id or `"auto"`. See [task-inference.md](task-inference.md) and [tasks.md](tasks.md). |
| `y_true` | Labels, targets, references, masks, box lists, or per-user relevant items — depends on task. |
| `y_pred` | Predictions, cluster labels, ranked lists, etc. |
| `y_prob` | Optional probability scores for classification / image classification / text classification. |
| `X` | Feature matrix for clustering (when not using label-only flow). |
| `timestamps` | Aligned with `y_true` / `y_pred` for time series. |
| `embeddings` | If set, inference may choose clustering. |
| `output_path` | Full path to the report file. Parent directory is created; also sets where plots go. |
| `format` | Output format if not obvious from `output_path` suffix. |
| `**kwargs` | Extra arguments for the underlying report (e.g. `k_values`, `labels`, `window` for rolling metrics). |

## Return value

Returns the same structure as `report.run_all()` / `report.to_dict()`:

```python
{
    "metrics": {...},
    "insights": ["...", ...],
    "plots": {"plot_key": "path/to.png", ...},
    "metric_descriptions": {"metric_name": "human text", ...},
}
```

## Output paths and plots

- **`output_path` set** — e.g. `my_reports/model.html`  
  - Report: `my_reports/model.html`  
  - Plots: `my_reports/evalreport_plots/*.png`

- **`output_path` omitted** — default directory is **`reports/`** (created under the current working directory).  
  - Report: `reports/<task>_report.<format>`  
  - Plots: `reports/evalreport_plots/`

The `format` argument controls the default filename extension when `output_path` is omitted.

## Explicit task names (quick map)

| Task family | Accepted `task` values (non-exhaustive) |
|-------------|----------------------------------------|
| Classification | `classification`, `binary_classification`, `multiclass`, `multilabel` |
| Regression | `regression` |
| Clustering | `clustering`, `cluster` |
| Time series | `timeseries`, `forecasting`, `time_series` |
| NLP | `text_classification`, `nlp_text_classification`, `text_generation`, `nlp_text_generation` |
| Vision | `segmentation`, `image_segmentation`, `detection`, `object_detection`, `image_classification`, `vision_classification` |
| Ranking | `ranking`, `recommendation`, `recommender` |

Full detail: [tasks.md](tasks.md).

## Errors

- Unknown `task` string → `ValueError` with a short list of supported tasks.
- Missing required inputs for a task → raised by the specific report class (e.g. time series without `timestamps`).
