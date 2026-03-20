from __future__ import annotations

from dataclasses import dataclass
from math import log2
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Set, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..core.base_report import BaseReport


def _to_set(x: Iterable[Any]) -> Set[Any]:
    return set(x)


def _to_list(x: Iterable[Any]) -> List[Any]:
    return list(x)


def precision_at_k(relevant: Set[Any], ranked: Sequence[Any], k: int) -> float:
    if k <= 0:
        return 0.0
    top = ranked[:k]
    if not top:
        return 0.0
    hits = sum(1 for item in top if item in relevant)
    return hits / float(k)


def recall_at_k(relevant: Set[Any], ranked: Sequence[Any], k: int) -> float:
    if not relevant:
        return 0.0
    top = ranked[:k]
    hits = sum(1 for item in top if item in relevant)
    return hits / float(len(relevant))


def hit_rate_at_k(relevant: Set[Any], ranked: Sequence[Any], k: int) -> float:
    if k <= 0:
        return 0.0
    top = ranked[:k]
    return 1.0 if any(item in relevant for item in top) else 0.0


def average_precision(relevant: Set[Any], ranked: Sequence[Any]) -> float:
    """Mean average precision for one user (binary relevance)."""
    if not relevant:
        return 0.0
    hits = 0
    prec_sum = 0.0
    for i, item in enumerate(ranked, start=1):
        if item in relevant:
            hits += 1
            prec_sum += hits / float(i)
    return prec_sum / float(len(relevant))


def dcg_at_k(relevant: Set[Any], ranked: Sequence[Any], k: int) -> float:
    dcg = 0.0
    for i, item in enumerate(ranked[:k], start=1):
        rel = 1.0 if item in relevant else 0.0
        dcg += rel / log2(i + 1)
    return dcg


def idcg_at_k(relevant: Set[Any], k: int) -> float:
    """Ideal DCG when all |R| relevant items are at the top (binary)."""
    n_rel = min(len(relevant), k)
    if n_rel == 0:
        return 0.0
    return sum(1.0 / log2(i + 1) for i in range(1, n_rel + 1))


def ndcg_at_k(relevant: Set[Any], ranked: Sequence[Any], k: int) -> float:
    ideal = idcg_at_k(relevant, k)
    if ideal <= 0:
        return 0.0
    return dcg_at_k(relevant, ranked, k) / ideal


@dataclass
class RankingReport(BaseReport):
    """Recommendation / ranking evaluation (list-wise per user or query).

    Parameters
    ----------
    relevant :
        Per-user (or per-query) ground-truth relevant items. Iterable of iterables.
    ranked :
        Per-user ranked recommendation lists (best first). Same length as ``relevant``.
    k_values :
        K cutoffs for P@K, R@K, NDCG@K, Hit@K.
    """

    relevant: Optional[Iterable[Iterable[Any]]] = None
    ranked: Optional[Iterable[Iterable[Any]]] = None
    k_values: Tuple[int, ...] = (1, 5, 10)

    def _normalize_inputs(self) -> Tuple[List[Set[Any]], List[List[Any]]]:
        if self.relevant is None or self.ranked is None:
            raise ValueError("RankingReport requires `relevant` and `ranked` (same number of users/queries).")
        rel_list = [_to_set(r) for r in self.relevant]
        rank_list = [_to_list(r) for r in self.ranked]
        if len(rel_list) != len(rank_list):
            raise ValueError("`relevant` and `ranked` must have the same length (one list per user/query).")
        if len(rel_list) == 0:
            raise ValueError("RankingReport requires at least one user/query.")
        return rel_list, rank_list

    def _compute_metrics(self) -> None:
        rel_list, rank_list = self._normalize_inputs()
        n_users = len(rel_list)

        # Per-K aggregates
        ks = sorted(set(max(1, int(k)) for k in self.k_values))
        max_k = max(ks)

        mean_p: dict[str, float] = {}
        mean_r: dict[str, float] = {}
        mean_ndcg: dict[str, float] = {}
        mean_hit: dict[str, float] = {}

        for k in ks:
            ps, rs, nds, hs = [], [], [], []
            for rel, ranked in zip(rel_list, rank_list):
                ps.append(precision_at_k(rel, ranked, k))
                rs.append(recall_at_k(rel, ranked, k))
                nds.append(ndcg_at_k(rel, ranked, k))
                hs.append(hit_rate_at_k(rel, ranked, k))
            mean_p[f"precision_at_{k}"] = float(np.mean(ps))
            mean_r[f"recall_at_{k}"] = float(np.mean(rs))
            mean_ndcg[f"ndcg_at_{k}"] = float(np.mean(nds))
            mean_hit[f"hit_rate_at_{k}"] = float(np.mean(hs))

        # MAP (mean of per-user AP)
        aps = [average_precision(rel, ranked) for rel, ranked in zip(rel_list, rank_list)]
        map_score = float(np.mean(aps))

        self.metrics.update(
            {
                "map": map_score,
                "num_users": n_users,
                **mean_p,
                **mean_r,
                **mean_ndcg,
                **mean_hit,
            }
        )

        # Descriptions for every metric key (HTML/PDF “Metric reference” section).
        self.metric_descriptions.update(
            {
                "map": (
                    "Mean Average Precision (MAP): average of per-user Average Precision, "
                    "where AP rewards placing relevant items higher in the ranked list (binary relevance per item)."
                ),
                "num_users": "Number of users or queries in the evaluation set (one ranked list per row).",
            }
        )
        for k in ks:
            self.metric_descriptions[f"precision_at_{k}"] = (
                f"Mean Precision@{k}: averaged across users—of the top-{k} recommendations, "
                f"what fraction are relevant (|relevant ∩ top-{k}| / {k})."
            )
            self.metric_descriptions[f"recall_at_{k}"] = (
                f"Mean Recall@{k}: averaged across users—what fraction of that user’s relevant items "
                f"appear in the top-{k} list (|relevant ∩ top-{k}| / |relevant|; 0 if no relevant items)."
            )
            self.metric_descriptions[f"ndcg_at_{k}"] = (
                f"Mean NDCG@{k}: averaged across users—normalized Discounted Cumulative Gain at rank {k}, "
                f"comparing the ranked list to an ideal that puts all relevant items first (binary relevance)."
            )
            self.metric_descriptions[f"hit_rate_at_{k}"] = (
                f"Mean Hit Rate@{k}: averaged across users—fraction of users who have at least one relevant item "
                f"in their top-{k} recommendations."
            )

        self._cached_ks = ks
        self._cached_max_k = max_k
        self._cached_rel = rel_list
        self._cached_rank = rank_list

    def _generate_plots(self) -> None:
        root = self.output_dir or Path("reports")
        plot_dir = root / "evalreport_plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        plots: dict[str, str] = {}

        ks = getattr(self, "_cached_ks", [])
        rel_list = getattr(self, "_cached_rel", None)
        rank_list = getattr(self, "_cached_rank", None)
        if not ks or rel_list is None or rank_list is None:
            self.plots = plots
            return

        # Precision@K curve
        try:
            precs = [self.metrics.get(f"precision_at_{k}", 0.0) for k in ks]
            plt.figure(figsize=(5.5, 4))
            plt.plot(ks, precs, marker="o", linewidth=2)
            plt.xlabel("K")
            plt.ylabel("Mean Precision@K")
            plt.title("Precision@K curve")
            plt.ylim(0, 1.05)
            plt.grid(True, alpha=0.3)
            path = plot_dir / "ranking_precision_at_k.png"
            plt.tight_layout()
            plt.savefig(path)
            plt.close()
            plots["precision_at_k_curve"] = str(path)
        except Exception:
            pass

        # Cumulative gain: mean cumulative relevant count vs rank (up to max_k)
        try:
            max_k = int(getattr(self, "_cached_max_k", max(ks)))
            cum = np.zeros(max_k, dtype=float)
            for rel, ranked in zip(rel_list, rank_list):
                for pos in range(max_k):
                    if pos < len(ranked) and ranked[pos] in rel:
                        cum[pos] += 1.0
            cum_mean = np.cumsum(cum) / float(len(rel_list))
            ranks = np.arange(1, max_k + 1)
            plt.figure(figsize=(5.5, 4))
            plt.plot(ranks, cum_mean, marker="s", linewidth=2, color="#2ca02c")
            plt.xlabel("Rank cutoff")
            plt.ylabel("Mean cumulative relevant retrieved")
            plt.title("Cumulative gain (mean over users)")
            plt.grid(True, alpha=0.3)
            path = plot_dir / "ranking_cumulative_gain.png"
            plt.tight_layout()
            plt.savefig(path)
            plt.close()
            plots["cumulative_gain"] = str(path)
        except Exception:
            pass

        self.plots = plots

    def _generate_insights(self) -> None:
        insights: List[str] = []
        ks = getattr(self, "_cached_ks", [])
        if len(ks) >= 2:
            k_small, k_large = ks[0], ks[-1]
            p_small = self.metrics.get(f"precision_at_{k_small}")
            p_large = self.metrics.get(f"precision_at_{k_large}")
            if isinstance(p_small, (int, float)) and isinstance(p_large, (int, float)):
                if p_large + 1e-6 < p_small * 0.85:
                    insights.append(
                        f"Precision@K drops from {k_small} to {k_large}; quality may degrade in deeper ranks."
                    )

        # Long-tail: users with many relevant items vs few
        rel_list = getattr(self, "_cached_rel", [])
        if rel_list:
            sizes = [len(r) for r in rel_list]
            if sizes:
                med = float(np.median(sizes))
                mx = max(sizes)
                if mx > 0 and med > 0 and mx / med >= 5:
                    insights.append(
                        "Large spread in number of relevant items per user; check long-tail users separately."
                    )

        map_score = self.metrics.get("map")
        if isinstance(map_score, (int, float)) and map_score < 0.1:
            insights.append("Low MAP; consider improving candidate generation, re-ranking, or relevance signals.")

        self.insights = insights
