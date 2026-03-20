#!/usr/bin/env python3
"""
Smoke-test model-eval-toolkit: run generate_report() for every supported task.

Uses larger, more realistic-style synthetic data (imbalanced classes, noisy series,
e-commerce style IDs, COCO-like boxes, etc.) while staying self-contained.

Usage (from project root):
    python tmp/run_all_tasks.py

Reports: tmp/reports/*.html and tmp/reports/evalreport_plots/
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Writable matplotlib config when ~/.matplotlib is not available (CI/sandbox).
ROOT = Path(__file__).resolve().parents[1]
_mpl_dir = ROOT / "tmp" / ".matplotlib"
_mpl_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_dir))
# Quieter sklearn/KMeans on some macOS / CI environments
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import numpy as np

# Project root on path for `import evalreport` without pip install
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "tmp" / "reports"
OUT.mkdir(parents=True, exist_ok=True)

from evalreport import generate_report  # noqa: E402


def _softmax_rows(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def main() -> None:
    rng = np.random.default_rng(42)
    print(f"Writing reports under: {OUT}\n")

    # --- 1) Classification: multiclass, imbalanced (e.g. support ticket routing) ---
    n = 180
    class_names = ["billing", "shipping", "refund", "other"]
    classes = np.array(class_names)
    # Column order for y_prob must match `labels` (sklearn expects consistent ordering)
    label_order = sorted(class_names)
    # Imbalanced true labels
    weights = np.array([0.45, 0.30, 0.15, 0.10])
    y_true_cls = rng.choice(classes, size=n, p=weights)
    # Noisy predictions: mostly correct, some confusions between billing/refund
    y_pred_cls = y_true_cls.copy()
    flip = rng.random(n) < 0.22
    y_pred_cls[flip] = rng.choice(classes, size=flip.sum())
    # Logits then softmax → realistic prob matrix (n, 4), columns = label_order
    col_index = {c: j for j, c in enumerate(label_order)}
    logits = rng.normal(0, 1, (n, len(label_order)))
    for i, t in enumerate(y_true_cls):
        logits[i, col_index[str(t)]] += 1.6
    y_prob_cls = _softmax_rows(logits)

    generate_report(
        task="classification",
        y_true=y_true_cls.tolist(),
        y_pred=y_pred_cls.tolist(),
        y_prob=y_prob_cls,
        labels=label_order,
        output_path=str(OUT / "classification.html"),
        format="html",
    )
    print("OK classification (multiclass, imbalanced)")

    # --- 2) Regression: house-style prices ($100k–$500k) with heteroskedastic noise ---
    n_reg = 200
    sqft = rng.uniform(800, 3200, n_reg)
    age = rng.integers(0, 45, n_reg)
    y_true_price = 50_000 + 120 * sqft - 800 * age + rng.normal(0, 12_000, n_reg)
    y_pred_price = y_true_price + rng.normal(0, 18_000, n_reg) + 5_000 * np.sin(sqft / 900)

    generate_report(
        task="regression",
        y_true=y_true_price.tolist(),
        y_pred=y_pred_price.tolist(),
        output_path=str(OUT / "regression.html"),
    )
    print("OK regression (noisy price-like targets)")

    # --- 3) Clustering: 2D embeddings, KMeans assignments (customer segments) ---
    from sklearn.cluster import KMeans
    from sklearn.datasets import make_blobs

    X_blob, _ = make_blobs(n_samples=400, centers=5, cluster_std=0.85, random_state=42)
    km = KMeans(n_clusters=5, random_state=42, n_init=10)
    labels_km = km.fit_predict(X_blob)

    generate_report(
        task="clustering",
        X=X_blob,
        y_pred=labels_km.tolist(),
        output_path=str(OUT / "clustering.html"),
    )
    print("OK clustering (blobs + KMeans labels)")

    # --- 4) Time series: daily sales with weekly seasonality + trend + forecast error ---
    days = np.arange(120)
    trend = 2.0 * days
    weekly = 25 * np.sin(2 * np.pi * days / 7)
    noise = rng.normal(0, 8, len(days))
    y_sales = 200 + trend + weekly + noise
    # Forecast: good early, slight drift / phase error later (realistic)
    y_fc = y_sales + rng.normal(0, 5, len(days))
    y_fc[60:] += rng.normal(2.5, 3, len(days) - 60)  # mild positive bias in 2nd half

    generate_report(
        task="timeseries",
        y_true=y_sales.tolist(),
        y_pred=y_fc.tolist(),
        timestamps=days.tolist(),
        output_path=str(OUT / "timeseries.html"),
        rolling_window=14,
    )
    print("OK timeseries (daily seasonality + drift)")

    # --- 5) Text classification: product review sentiment (short labels) ---
    sentiments = ["positive", "negative", "neutral"]
    n_txt = 120
    y_true_sent = rng.choice(sentiments, size=n_txt, p=[0.42, 0.33, 0.25])
    y_pred_sent = y_true_sent.copy()
    mis = rng.random(n_txt) < 0.18
    y_pred_sent[mis] = rng.choice(sentiments, size=mis.sum())

    generate_report(
        task="text_classification",
        y_true=y_true_sent.tolist(),
        y_pred=y_pred_sent.tolist(),
        output_path=str(OUT / "text_classification.html"),
    )
    print("OK text_classification (imbalanced sentiment)")

    # --- 6) Text generation: short “summary” style references vs model outputs ---
    refs = [
        "The quarterly revenue grew twelve percent year over year driven by enterprise subscriptions.",
        "The FDA approved the drug for adults with moderate plaque psoriasis after phase three trials.",
        "Shipping delays were caused by port congestion on the west coast during the holiday season.",
        "The company will cut carbon emissions forty percent by twenty thirty using renewable energy.",
        "Users can enable two factor authentication in the security tab of account settings.",
        "Inflation cooled to three point one percent as energy prices declined from summer peaks.",
        "The patch fixes a memory leak in the worker process when handling large file uploads.",
        "Restaurant same store sales rose five percent compared to the prior year quarter.",
    ]
    hyps = [
        "Quarterly revenue increased twelve percent YoY due to strong enterprise subscription sales.",
        "FDA approved the treatment for adults with moderate plaque psoriasis following phase 3 trials.",
        "Delays happened because of west coast port congestion during holidays.",
        "Company plans to reduce emissions by forty percent by 2030 with renewables.",
        "Two factor authentication is available under security in account settings.",
        "Inflation fell to three point one percent as energy costs dropped after summer.",
        "This update fixes a worker memory leak when processing big uploads.",
        "Same store restaurant sales were up five percent versus last year quarter.",
    ]
    generate_report(
        task="text_generation",
        y_true=refs,
        y_pred=hyps,
        output_path=str(OUT / "text_generation.html"),
    )
    print("OK text_generation (news / product style sentences)")

    # --- 7) Segmentation: small “medical / cell” style masks (4 samples, 64×64) ---
    h, w = 64, 64
    n_seg = 4
    yt = np.zeros((n_seg, h, w), dtype=np.uint8)
    yp = np.zeros((n_seg, h, w), dtype=np.uint8)
    for s in range(n_seg):
        cy, cx = rng.integers(18, 46, 2)
        rr = rng.integers(10, 18)
        yy, xx = np.ogrid[:h, :w]
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= rr**2
        yt[s] = mask.astype(np.uint8)
        # Prediction: slightly shifted / eroded blob
        shift_y, shift_x = rng.integers(-3, 4, 2)
        yy2, xx2 = np.ogrid[:h, :w]
        mask_p = (yy2 - cy - shift_y) ** 2 + (xx2 - cx - shift_x) ** 2 <= (rr - rng.integers(0, 3)) ** 2
        yp[s] = mask_p.astype(np.uint8)

    generate_report(
        task="segmentation",
        y_true=yt,
        y_pred=yp,
        output_path=str(OUT / "segmentation.html"),
        max_visualizations=4,
    )
    print("OK segmentation (64×64 blob masks)")

    # --- 8) Detection: multi-image, COCO-like categories + scores ---
    images_gt = []
    images_pr = []
    for _ in range(6):
        n_obj = rng.integers(1, 4)
        gt_boxes = []
        pr_boxes = []
        for _o in range(n_obj):
            x1, y1 = rng.integers(20, 280, 2)
            bw, bh = rng.integers(40, 120, 2)
            x2, y2 = min(640, x1 + bw), min(480, y1 + bh)
            lab = rng.choice(["person", "car", "dog", "bicycle"])
            gt_boxes.append({"bbox": [float(x1), float(y1), float(x2), float(y2)], "label": lab})
            # Prediction: small jitter
            j = rng.normal(0, 4, 4)
            px1, py1, px2, py2 = x1 + j[0], y1 + j[1], x2 + j[2], y2 + j[3]
            px1, py1 = max(0, px1), max(0, py1)
            px2, py2 = max(px1 + 5, px2), max(py1 + 5, py2)
            sc = float(rng.uniform(0.35, 0.98))
            pr_boxes.append(
                {"bbox": [float(px1), float(py1), float(px2), float(py2)], "label": lab, "score": sc}
            )
        # occasional extra false positive
        if rng.random() < 0.35:
            pr_boxes.append(
                {
                    "bbox": [400.0, 300.0, 500.0, 400.0],
                    "label": "person",
                    "score": float(rng.uniform(0.25, 0.55)),
                }
            )
        images_gt.append(gt_boxes)
        images_pr.append(pr_boxes)

    generate_report(
        task="detection",
        y_true=images_gt,
        y_pred=images_pr,
        output_path=str(OUT / "detection.html"),
        max_visualizations=3,
    )
    print("OK detection (multi-image, multi-class boxes)")

    # --- 9) Image classification: ImageNet-style class ids, holdout batch ---
    n_img = 96
    n_classes = 10  # subset of “classes”
    y_true_img = rng.integers(0, n_classes, size=n_img)
    y_pred_img = y_true_img.copy()
    wrong = rng.random(n_img) < 0.16
    y_pred_img[wrong] = rng.integers(0, n_classes, size=wrong.sum())
    logits_img = rng.normal(0, 1, (n_img, n_classes))
    logits_img[np.arange(n_img), y_true_img] += 1.4
    y_prob_img = _softmax_rows(logits_img)

    generate_report(
        task="image_classification",
        y_true=y_true_img.tolist(),
        y_pred=y_pred_img.tolist(),
        y_prob=y_prob_img,
        labels=list(range(n_classes)),
        output_path=str(OUT / "image_classification.html"),
    )
    print("OK image_classification (multiclass probs, 96 “images”)")

    # --- 10) Recommendation: SKU-style items, many users, long-tail relevance ---
    n_users = 80
    catalog = [f"SKU_{i:05d}" for i in range(500)]
    relevant_per_user = []
    ranked_per_user = []
    for _u in range(n_users):
        n_rel = int(rng.integers(1, 12))  # 1–11 relevant
        rel = set(rng.choice(catalog, size=n_rel, replace=False).tolist())
        # Ranked list: mix hits + random exploration (80 items)
        ranked = []
        pool = list(rel) + rng.choice(catalog, size=100, replace=True).tolist()
        rng.shuffle(pool)
        seen = set()
        for item in pool:
            if item not in seen:
                seen.add(item)
                ranked.append(item)
            if len(ranked) >= 80:
                break
        relevant_per_user.append(list(rel))
        ranked_per_user.append(ranked)

    generate_report(
        task="recommendation",
        y_true=relevant_per_user,
        y_pred=ranked_per_user,
        k_values=(1, 5, 10, 20),
        output_path=str(OUT / "recommendation.html"),
    )
    print("OK recommendation (80 users, SKU catalog, K=1,5,10,20)")

    print(f"\nDone. Open HTML files in: {OUT}")


if __name__ == "__main__":
    main()
