# Supported tasks

Each task has a dedicated report class under `evalreport`. All share: `run_all()`, `to_dict()`, `save(path, format=...)`, and optional `output_dir` for plot locations.

Use **`generate_report(...)`** or construct the class directly for full control.

---

## Classification

**`task`:** `classification`, `binary_classification`, `multiclass`, `multilabel`  
**Class:** `ClassificationReport`

**Inputs**

- `y_true`, `y_pred`: sequences of labels (int, str, or bool).
- `y_prob` (optional):  
  - Binary: length-`n` scores for positive class, or shape `(n, 2)`.  
  - Multiclass: `(n_samples, n_classes)` for log loss / ROC-PR (OvR) where applicable.
- `labels` (optional): fixed class order for confusion matrix / metrics.

**Metrics (high level)**  
Accuracy; precision / recall / F1 (micro, macro, weighted); MCC; Cohen’s κ; log loss (with probs); ROC-AUC / PR-AUC when applicable; confusion matrix counts.

**Plots**  
Confusion matrix heatmap; binary or multiclass ROC/PR when probabilities are available.

**Insights**  
Class imbalance hints; common misclassification pairs.

```python
from evalreport import generate_report

generate_report(
    task="classification",
    y_true=[0, 1, 2, 0],
    y_pred=[0, 2, 2, 0],
    y_prob=[[0.7, 0.2, 0.1], [0.1, 0.2, 0.7], ...],  # optional
    labels=[0, 1, 2],
    output_path="reports/classification_report.html",
)
```

---

## Regression

**`task`:** `regression`  
**Class:** `RegressionReport`

**Inputs:** `y_true`, `y_pred` (float sequences).

**Metrics:** MAE, MSE, RMSE, R², median absolute error, MAPE (where defined), mean error (bias).

**Plots:** Residuals vs predicted, predicted vs actual, residual histogram.

**Insights:** Over/under-prediction bias; heavy-tail error hints.

```python
generate_report(
    task="regression",
    y_true=[1.0, 2.0, 3.0],
    y_pred=[1.1, 1.9, 3.2],
    output_path="reports/regression_report.html",
)
```

---

## Clustering

**`task`:** `clustering`, `cluster`  
**Class:** `ClusteringReport`

**Inputs**

- `X`: feature matrix (2D array-like).
- `labels`: cluster assignment per sample — pass as `y_pred` to `generate_report` (or `labels=` on the class).

**Metrics:** Silhouette (when valid), Davies–Bouldin, Calinski–Harabasz, cluster size stats.

**Plots:** 2D PCA scatter colored by cluster; cluster size distribution.

**Insights:** Separability and cluster balance hints.

```python
generate_report(
    task="clustering",
    X=[[0, 0], [0.1, 0.2], [5, 5], [5.1, 4.9]],
    y_pred=[0, 0, 1, 1],
    output_path="reports/clustering_report.html",
)
```

---

## Time series / forecasting

**`task`:** `timeseries`, `forecasting`, `time_series`  
**Class:** `TimeSeriesReport`

**Inputs:** `y_true`, `y_pred`, **`timestamps`** (same length).

**Metrics:** MAE, MSE, RMSE, MAPE, SMAPE, mean forecast error, rolling RMSE summary.

**Plots:** Actual vs forecast, residuals over time, rolling RMSE.

**Insights:** Bias and stability / drift hints from rolling metrics.

```python
generate_report(
    task="timeseries",
    y_true=[1, 2, 3, 4],
    y_pred=[1.1, 1.9, 3.2, 3.8],
    timestamps=["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
    output_path="reports/timeseries_report.html",
)
```

---

## NLP — text classification

**`task`:** `text_classification`, `nlp_text_classification`  
**Class:** `TextClassificationReport`

**Inputs:** `y_true`, `y_pred` as string labels; optional `y_prob` analogous to tabular classification.

**Metrics / plots / insights:** Reuses **`ClassificationReport`** logic (same metrics, confusion matrix, ROC/PR when probabilities are provided).

```python
generate_report(
    task="text_classification",
    y_true=["spam", "ham", "spam"],
    y_pred=["spam", "spam", "spam"],
    output_path="reports/nlp_cls_report.html",
)
```

---

## NLP — text generation

**`task`:** `text_generation`, `nlp_text_generation`  
**Class:** `TextGenerationReport`

**Inputs:** `references` / `y_true` and `predictions` / `y_pred` as parallel sequences of strings.

**Metrics:** `bleu_like`, `rouge_l_f1_like`, `token_jaccard` (mean), `num_samples`.

**Plots:** Histogram of per-sample token Jaccard overlap.

```python
generate_report(
    task="text_generation",
    y_true=["the cat sat on the mat"],
    y_pred=["the cat sat on mat"],
    output_path="reports/nlp_gen_report.html",
)
```

---

## Vision — image classification

**`task`:** `image_classification`, `vision_classification`  
**Class:** `ImageClassificationReport`

**Inputs:** `y_true`, `y_pred` (class labels per image); optional `y_prob`.

**Metrics / plots:** Classification-style metrics and confusion matrix for image labels.

```python
generate_report(
    task="image_classification",
    y_true=["cat", "dog", "cat"],
    y_pred=["cat", "cat", "cat"],
    output_path="reports/imgcls_report.html",
)
```

---

## Vision — segmentation

**`task`:** `segmentation`, `image_segmentation`  
**Class:** `SegmentationReport`

**Inputs:** `y_true_masks`, `y_pred_masks` as array-like (per-image binary or label masks). With `generate_report`, pass as `y_true` / `y_pred`.

**Metrics:** Pixel / region overlap style metrics (IoU-related).

**Plots:** Visual comparisons where applicable.

```python
import numpy as np
from evalreport import generate_report

generate_report(
    task="segmentation",
    y_true=[np.array([[0, 1], [1, 1]])],
    y_pred=[np.array([[0, 1], [0, 1]])],
    output_path="reports/segmentation_report.html",
)
```

---

## Vision — object detection

**`task`:** `detection`, `object_detection`  
**Class:** `DetectionReport`

**Inputs:** Per-image lists of dicts. Ground truth and predictions should include **`bbox`** (e.g. `[x1, y1, x2, y2]`). Predictions typically include **`score`** and **`label`** for matching.

**Metrics:** COCO-style mAP @ IoU 0.50:0.95 and per-threshold summaries.

**Plots:** Summary visualizations for detection quality.

```python
generate_report(
    task="detection",
    y_true=[[{"bbox": [0, 0, 10, 10], "label": "obj"}]],
    y_pred=[[{"bbox": [1, 1, 9, 9], "label": "obj", "score": 0.9}]],
    output_path="reports/detection_report.html",
)
```

---

## Recommendation / ranking

**`task`:** `ranking`, `recommendation`, `recommender`  
**Class:** `RankingReport`

**Inputs**

- `relevant` / `y_true`: list of **relevant item IDs** per user (each element is a list/tuple/set).
- `ranked` / `y_pred`: **ordered** recommended item IDs per user (best rank first).
- `k_values` (optional): tuple of cutoffs, default `(1, 5, 10)`.

**Metrics:** MAP, Precision@K, Recall@K, NDCG@K, Hit Rate@K.

**Plots:** Precision@K curve; mean cumulative gain vs cutoff.

```python
generate_report(
    task="recommendation",
    y_true=[[10, 20], [30]],
    y_pred=[[10, 99, 20, 5], [7, 30]],
    k_values=(1, 5, 10),
    output_path="reports/ranking_report.html",
)
```

---

## Roadmap

Multilabel enhancements, richer recommender scenarios (sessions, implicit feedback), and a plugin-style API may land in future versions. See GitHub issues for discussion.
