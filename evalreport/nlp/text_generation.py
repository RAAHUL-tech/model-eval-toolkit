from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..core.base_report import BaseReport


def _as_str_list(x: Optional[Iterable[Any]]) -> Optional[List[str]]:
    if x is None:
        return None
    return ["" if v is None else str(v) for v in x]


def _tokenize(text: str) -> List[str]:
    # Minimal tokenizer: whitespace split. Keeps v0.1 dependency-light.
    return [t for t in text.strip().split() if t]


def _ngram_counts(tokens: List[str], n: int) -> dict[tuple[str, ...], int]:
    counts: dict[tuple[str, ...], int] = {}
    if n <= 0:
        return counts
    for i in range(0, max(0, len(tokens) - n + 1)):
        ng = tuple(tokens[i : i + n])
        counts[ng] = counts.get(ng, 0) + 1
    return counts


def _bleu_like(references: Sequence[str], predictions: Sequence[str], max_n: int = 4) -> float:
    """A lightweight BLEU-like score (not sacreBLEU).

    Computes clipped n-gram precisions with a brevity penalty and uniform weights.
    Intended as a reasonable v0.1 metric without extra dependencies.
    """
    # corpus n-gram stats
    clipped_hits = np.zeros(max_n, dtype=float)
    total_preds = np.zeros(max_n, dtype=float)
    ref_len = 0
    pred_len = 0

    for ref, pred in zip(references, predictions):
        ref_toks = _tokenize(ref)
        pred_toks = _tokenize(pred)
        ref_len += len(ref_toks)
        pred_len += len(pred_toks)
        for n in range(1, max_n + 1):
            ref_counts = _ngram_counts(ref_toks, n)
            pred_counts = _ngram_counts(pred_toks, n)
            total_preds[n - 1] += sum(pred_counts.values())
            for ng, c in pred_counts.items():
                clipped_hits[n - 1] += min(c, ref_counts.get(ng, 0))

    precisions = []
    for n in range(max_n):
        if total_preds[n] == 0:
            precisions.append(0.0)
        else:
            precisions.append(float(clipped_hits[n] / total_preds[n]))

    # geometric mean with smoothing for zeros
    eps = 1e-12
    log_p = np.mean([np.log(max(p, eps)) for p in precisions])
    geo_mean = float(np.exp(log_p))

    # brevity penalty
    if pred_len == 0:
        return 0.0
    bp = 1.0 if pred_len > ref_len else float(np.exp(1.0 - (ref_len / max(1, pred_len))))
    return float(bp * geo_mean)


def _rouge_l_like(references: Sequence[str], predictions: Sequence[str]) -> float:
    """Lightweight ROUGE-L F1 over whitespace tokens (corpus average)."""

    def lcs(a: List[str], b: List[str]) -> int:
        # DP LCS length
        dp = np.zeros((len(a) + 1, len(b) + 1), dtype=int)
        for i in range(1, len(a) + 1):
            for j in range(1, len(b) + 1):
                if a[i - 1] == b[j - 1]:
                    dp[i, j] = dp[i - 1, j - 1] + 1
                else:
                    dp[i, j] = max(dp[i - 1, j], dp[i, j - 1])
        return int(dp[len(a), len(b)])

    scores = []
    for ref, pred in zip(references, predictions):
        r = _tokenize(ref)
        p = _tokenize(pred)
        if not r and not p:
            scores.append(1.0)
            continue
        if not r or not p:
            scores.append(0.0)
            continue
        l = lcs(r, p)
        prec = l / max(1, len(p))
        rec = l / max(1, len(r))
        if prec + rec == 0:
            scores.append(0.0)
        else:
            scores.append(2 * prec * rec / (prec + rec))
    return float(np.mean(scores)) if scores else 0.0


@dataclass
class TextGenerationReport(BaseReport):
    references: Optional[Iterable[Any]] = None
    predictions: Optional[Iterable[Any]] = None

    def _compute_metrics(self) -> None:
        refs = _as_str_list(self.references)
        preds = _as_str_list(self.predictions)
        if refs is None or preds is None:
            raise ValueError("TextGenerationReport requires references and predictions.")
        if len(refs) != len(preds):
            raise ValueError("TextGenerationReport requires references and predictions of equal length.")

        bleu = _bleu_like(refs, preds)
        rouge_l = _rouge_l_like(refs, preds)

        # Simple lexical overlap ratio (Jaccard over tokens) as a semantic-gap proxy
        overlaps = []
        for r, p in zip(refs, preds):
            rt = set(_tokenize(r))
            pt = set(_tokenize(p))
            denom = len(rt | pt)
            overlaps.append(1.0 if denom == 0 else (len(rt & pt) / denom))
        jaccard = float(np.mean(overlaps)) if overlaps else 0.0

        self.metrics.update(
            {
                "bleu_like": float(bleu),
                "rouge_l_f1_like": float(rouge_l),
                "token_jaccard": float(jaccard),
                "num_samples": int(len(refs)),
            }
        )

        self.metric_descriptions.update(
            {
                "bleu_like": "N-gram overlap score with brevity penalty (lightweight BLEU-like). Higher is better.",
                "rouge_l_f1_like": "Sequence overlap based on LCS (lightweight ROUGE-L F1). Higher is better.",
                "token_jaccard": "Token-set Jaccard overlap between prediction and reference (lexical overlap proxy).",
                "num_samples": "Number of evaluated reference/prediction pairs.",
            }
        )

        self._cached_scores = {
            "bleu_like": [bleu],  # corpus-level; keep structure for plots
            "rouge_l_f1_like": [rouge_l],
            "token_jaccard": overlaps,
        }

    def _generate_plots(self) -> None:
        root = self.output_dir or Path("reports")
        plot_dir = root / "evalreport_plots"
        plot_dir.mkdir(parents=True, exist_ok=True)

        plots: dict[str, str] = {}

        # Distribution of token_jaccard per sample
        try:
            vals = np.asarray(self._cached_scores.get("token_jaccard", []), dtype=float)
            if vals.size > 0:
                plt.figure(figsize=(5, 3.5))
                plt.hist(vals, bins=20, alpha=0.85)
                plt.xlabel("Token Jaccard overlap")
                plt.ylabel("Count")
                plt.title("Generation overlap distribution")
                path = plot_dir / "nlp_generation_overlap_distribution.png"
                plt.tight_layout()
                plt.savefig(path)
                plt.close()
                plots["overlap_distribution"] = str(path)
        except Exception:
            pass

        self.plots = plots

    def _generate_insights(self) -> None:
        insights: List[str] = []
        j = self.metrics.get("token_jaccard")
        b = self.metrics.get("bleu_like")
        r = self.metrics.get("rouge_l_f1_like")

        if isinstance(j, (int, float)) and j < 0.2:
            insights.append("Low lexical overlap; model outputs may be paraphrasing heavily or drifting semantically.")
        if isinstance(b, (int, float)) and b < 0.1 and isinstance(r, (int, float)) and r < 0.2:
            insights.append("Very low overlap-based scores; check formatting, tokenization, or reference/prediction alignment.")

        self.insights = insights

