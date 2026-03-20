# Getting started

## Install

```bash
pip install model-eval-toolkit
```

**Requirements:** Python ≥ 3.9, NumPy, pandas, scikit-learn, Matplotlib, Seaborn (installed automatically).

## Optional extras

| Extra | Purpose |
|-------|---------|
| `pdf` | PDF export via ReportLab |
| `test` | pytest + coverage (for contributors) |
| `nlp` | Reserved for future NLP-only deps (currently empty) |
| `vision` | Reserved for future vision-only deps (currently empty) |

```bash
pip install "model-eval-toolkit[pdf]"
pip install "model-eval-toolkit[test,pdf]"
```

## Import name

The distribution name on PyPI is **`model-eval-toolkit`**. You import from **`evalreport`**:

```python
from evalreport import generate_report, __version__
```

## First report

```python
from evalreport import generate_report

summary = generate_report(
    task="regression",
    y_true=[1.0, 2.0, 3.0],
    y_pred=[1.1, 1.9, 3.2],
    output_path="reports/regression_report.html",
)

print(summary["metrics"]["rmse"])
```

If you omit `output_path`, files go under `reports/` by default. See [generate-report.md](generate-report.md).

## Next steps

- [generate-report.md](generate-report.md) — full API for `generate_report`
- [tasks.md](tasks.md) — inputs and examples per task type
- [task-inference.md](task-inference.md) — using `task="auto"`
