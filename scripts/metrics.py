"""Ranking and beyond-accuracy metrics.

Accuracy metrics alone reward recommending the ten most popular films to
everyone — a system that scores well and serves users badly. Coverage, novelty
and intra-list diversity are computed alongside so the tradeoff is visible
rather than hidden (README section 10).

All functions take `ranked` as item indices in predicted order, and `relevant`
as the set of held-out item indices the user actually liked.
"""

from __future__ import annotations

import numpy as np


def precision_at_k(ranked: np.ndarray, relevant: set[int], k: int) -> float:
    if k == 0:
        return 0.0
    hits = sum(1 for item in ranked[:k] if item in relevant)
    return hits / k


def recall_at_k(ranked: np.ndarray, relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for item in ranked[:k] if item in relevant)
    return hits / len(relevant)


def average_precision_at_k(ranked: np.ndarray, relevant: set[int], k: int) -> float:
    """AP@k — precision averaged at each hit position.

    Normalised by min(|relevant|, k) rather than |relevant|, so a user with 50
    relevant items is not automatically capped at 0.2 when k = 10.
    """
    if not relevant:
        return 0.0
    hits = 0
    total = 0.0
    for rank, item in enumerate(ranked[:k], start=1):
        if item in relevant:
            hits += 1
            total += hits / rank
    denominator = min(len(relevant), k)
    return total / denominator if denominator else 0.0


def ndcg_at_k(ranked: np.ndarray, relevant: set[int], k: int) -> float:
    """Binary-gain NDCG@k.

    Binary rather than graded: relevance here is "rated >= 4.0", which is
    already a thresholded judgement. Inventing graded gains from the star value
    would imply a precision the threshold has thrown away.
    """
    if not relevant:
        return 0.0
    gains = np.array([1.0 if item in relevant else 0.0 for item in ranked[:k]])
    discounts = 1.0 / np.log2(np.arange(2, len(gains) + 2))
    dcg = float(np.sum(gains * discounts))

    ideal_hits = min(len(relevant), k)
    idcg = float(np.sum(1.0 / np.log2(np.arange(2, ideal_hits + 2))))
    return dcg / idcg if idcg > 0 else 0.0


def intra_list_diversity(ranked: np.ndarray, embeddings: np.ndarray, k: int) -> float:
    """1 - mean pairwise cosine similarity across the slate.

    Embeddings are L2-normalised, so the dot product is the cosine. A slate of
    near-identical films scores near 0; an eclectic one approaches 1.
    """
    items = ranked[:k]
    if len(items) < 2:
        return 0.0
    vectors = embeddings[items]
    similarity = vectors @ vectors.T
    upper = similarity[np.triu_indices(len(items), k=1)]
    return float(1.0 - upper.mean())


def novelty(ranked: np.ndarray, item_novelty: np.ndarray, k: int) -> float:
    """Mean self-information of the slate: high means obscure recommendations."""
    items = ranked[:k]
    return float(item_novelty[items].mean()) if len(items) else 0.0


def catalogue_coverage(all_ranked: list[np.ndarray], n_items: int, k: int) -> float:
    """Share of the catalogue that appears in any user's top-k, as a percent."""
    surfaced: set[int] = set()
    for ranked in all_ranked:
        surfaced.update(int(i) for i in ranked[:k])
    return len(surfaced) / n_items * 100


def gini(all_ranked: list[np.ndarray], n_items: int, k: int) -> float:
    """Gini coefficient of exposure across items.

    0 = every item recommended equally often, 1 = all exposure on one item.
    Coverage says *how many* items are reached; this says how evenly. A system
    can cover 30% of the catalogue while sending 90% of impressions to ten
    films, and only this metric catches that.
    """
    counts = np.zeros(n_items, dtype=np.float64)
    for ranked in all_ranked:
        for item in ranked[:k]:
            counts[item] += 1
    if counts.sum() == 0:
        return 0.0
    sorted_counts = np.sort(counts)
    index = np.arange(1, n_items + 1)
    return float((2 * index - n_items - 1) @ sorted_counts / (n_items * sorted_counts.sum()))
