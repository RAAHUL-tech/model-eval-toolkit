# model-eval-toolkit

**Unified ML evaluation reports** for Python: metrics, plots, auto-insights, and export to **HTML**, **JSON**, **Markdown**, or **PDF**.

Import from the **`evalreport`** package:

```python
from evalreport import (
    generate_report,
    ClassificationReport,
    RegressionReport,
    ClusteringReport,
    TimeSeriesReport,
    __version__,
)
```

> **Current supported tasks (v0.1):** **Classification** (binary & multiclass), **regression**, **clustering**, and **time series**.
> The roadmap includes ranking/recsys, NLP, CV, and multilabel.

---

## Install

```bash
pip install model-eval-toolkit
```

**PDF export** needs ReportLab:

```bash
pip install "model-eval-toolkit[pdf]"
# or
pip install reportlab
```

**Requirements:** Python ≥ 3.9, NumPy, pandas, scikit-learn, Matplotlib, Seaborn.

---

## Quick start

### `generate_report` (recommended)

```python
from evalreport import generate_report

summary = generate_report(
    task="classification",  # or "regression", or "auto"
    y_true=[0, 1, 0, 1, 1],
    y_pred=[0, 1, 1, 1, 1],
    y_prob=[0.1, 0.9, 0.8, 0.7, 0.6],  # optional; enables log loss, ROC/PR (binary)
    output_path="my_reports/model_report.html",
    format="html",
)

print(summary["metrics"]["accuracy"])
```

- **`task="auto"`** — float targets → regression; integer/string labels → classification.
- If you **omit `output_path`**, the report is written under **`reports/`** (created if needed), e.g. `reports/classification_report.html` or `reports/regression_report.json` when `format="json"`.
- **Plots** are saved under **`<report_directory>/evalreport_plots/`** (same folder as your HTML/JSON/PDF file’s parent). So custom `output_path="my_reports/x.html"` → plots in `my_reports/evalreport_plots/`.

### Task-specific API

Useful when you want full control (e.g. set `output_dir` before `run_all()` so plots land next to a chosen folder):

```python
from pathlib import Path
from evalreport import ClassificationReport, RegressionReport

# Classification (binary or multiclass)
cls = ClassificationReport(
    y_true=[0, 1, 2, 0],
    y_pred=[0, 2, 2, 0],
    # y_prob: (n_samples, n_classes) for multiclass log loss / AUC
    labels=[0, 1, 2],  # optional fixed class order for confusion matrix
)
cls.output_dir = Path("reports")  # optional; default for plots if set before run_all()
cls.run_all()
cls.save("reports/classification_report.html", format="html")
cls.save("reports/classification_report.json", format="json")

# Regression
reg = RegressionReport(y_true=[1.0, 2.0, 3.0], y_pred=[1.1, 1.9, 3.2])
reg.output_dir = Path("reports")
reg.run_all()
reg.save("reports/regression_report.pdf", format="pdf")  # needs reportlab
```

---

## What each task includes

### Classification

| Area | Details |
|------|--------|
| **Metrics** | Accuracy; precision / recall / F1 (micro, macro, weighted); MCC; Cohen’s κ; log loss (with probs); ROC-AUC / PR-AUC when applicable; confusion matrix (table). |
| **Plots** | Confusion matrix heatmap; **binary** ROC & PR curves when `y_prob` is provided. |
| **Insights** | Class imbalance hint; most common misclassification pair. |
| **HTML** | Styled layout: each metric with a short explanation, insights, and embedded plot images. |

**Probabilities**

- Binary: `y_prob` as length-`n` scores for the positive class, or shape `(n, 2)`.
- Multiclass: `(n_samples, n_classes)` for log loss / multiclass AUC where supported.

### Regression

| Area | Details |
|------|--------|
| **Metrics** | MAE, MSE, RMSE, R², median absolute error, MAPE (where defined), mean error (bias). |
| **Plots** | Residuals vs predicted, predicted vs actual, residual histogram. |
| **Insights** | Over/under-prediction bias; heavy-tail error hint. |
| **HTML** | Same rich layout as classification. |

### Clustering

| Area | Details |
|------|--------|
| **Inputs** | `X` (feature matrix) and `labels` (cluster assignments) |
| **Metrics** | Silhouette score, Davies–Bouldin index, Calinski–Harabasz score, cluster sizes |
| **Plots** | Cluster scatter (PCA) and cluster size distribution |
| **Insights** | Separability + imbalance hints |
| **HTML** | Styled metrics/insights plus embedded plot images |

### Time Series / Forecasting

| Area | Details |
|------|--------|
| **Inputs** | `y_true`, `y_pred`, and `timestamps` (same length) |
| **Metrics** | MAE, MSE, RMSE, MAPE, SMAPE, mean forecast error, rolling RMSE summary |
| **Plots** | Actual vs forecast, residuals over time, rolling RMSE over time |
| **Insights** | Systematic bias and drift/stability hints via rolling RMSE |
| **HTML** | Styled metrics/insights plus embedded plot images |

---

## Output formats

| Format | How | Notes |
|--------|-----|--------|
| **HTML** | `format="html"` or `.html` | Metrics + descriptions + insights + plot images. |
| **JSON** | `format="json"` or `.json` | `metrics`, `insights`, `plots` (paths to PNGs). |
| **Markdown** | `format="markdown"` or `.md` | Metrics and insights (no embedded images). |
| **PDF** | `format="pdf"` or `.pdf` | Text summary (metrics + descriptions + insights); install `reportlab`. |

---

## Development

```bash
git clone https://github.com/RAAHUL-tech/model-eval-toolkit.git
cd model-eval-toolkit
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[test]"
pytest -q
# optional coverage
pytest --cov=evalreport --cov-report=term-missing
```

Build and check the package:

```bash
pip install build twine
python -m build
twine check dist/*
```

---

## Roadmap

Additional task types (clustering, time series, ranking, NLP, CV) and a plugin-style API are planned. Issues and PRs welcome on [GitHub](https://github.com/RAAHUL-tech/model-eval-toolkit).

---

## License

See [LICENSE](LICENSE).
