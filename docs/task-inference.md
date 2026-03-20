# Task inference (`task="auto"`)

When `task="auto"`, `generate_report` calls `evalreport.core.task_inference.infer_task` using the arguments you pass. This is **heuristic** — for production pipelines, prefer an explicit `task=` when you know the problem type.

## Order of checks (summary)

1. **`timestamps` provided** → **time series**
2. **`embeddings` provided** → **clustering** (hint path)
3. If **`y_true` and `y_pred`** are both set:
   - Structure matches **detection** (per-image lists of dicts with `bbox`) → **detection**
   - Arrays look like **2D+ masks** → **segmentation**
   - Nested lists like **relevant items + ranked lists per user** → **recommendation / ranking**
   - Both sides are **strings**: long sequences → **text generation**; short → **text classification**
4. **`X` is None**, **`y_pred` only`**, 1D discrete-like labels → **clustering**
5. **Regression vs classification**
   - Float targets: **regression** unless both `y_true` and `y_pred` look **integer-like** (then **classification**)
   - Non-float targets → **classification**
6. Fallback → **classification**

## Practical tips

- **Float targets that are “almost integers”** still go to **regression** if predictions are not integer-like — avoids misclassifying regression problems.
- **Recommendation** expects `y_true[i]` and `y_pred[i]` to be iterable per user (e.g. lists of item IDs).
- **Detection** expects each image’s boxes as dicts including **`bbox`** (e.g. `[x1, y1, x2, y2]`).
- When inference is wrong, set **`task=` explicitly** — no penalty.

## API

You can call inference directly (advanced):

```python
from evalreport.core.task_inference import infer_task

t = infer_task(y_true=..., y_pred=..., X=..., timestamps=..., embeddings=...)
```
