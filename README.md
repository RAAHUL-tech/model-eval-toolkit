# model-eval-toolkit
model-eval-toolkit is a unified Python library for generating comprehensive evaluation reports across machine learning tasks, including classification, regression, clustering, time series, NLP, computer vision, and recommendation systems with built-in metrics, visualizations, and exportable HTML/Markdown outputs.

## Install

```bash
pip install model-eval-toolkit
```

## Quick start (v0.1)

### Universal entry point

```python
from evalreport import generate_report

result = generate_report(
    task="classification",
    y_true=[0, 1, 0, 1],
    y_pred=[0, 1, 1, 1],
    output_path="report.html",
    format="html",
)

print(result["metrics"]["accuracy"])
```

### Task-specific APIs

```python
from evalreport import ClassificationReport, RegressionReport

cls = ClassificationReport(
    y_true=[0, 1, 0, 1],
    y_pred=[0, 1, 1, 1],
    y_prob=[0.1, 0.9, 0.7, 0.8],
)
cls.run_all()
cls.save("classification_report.html")

reg = RegressionReport(
    y_true=[1.0, 2.0, 3.0],
    y_pred=[1.0, 2.2, 2.7],
)
reg.run_all()
reg.save("regression_report.json", format="json")
```
