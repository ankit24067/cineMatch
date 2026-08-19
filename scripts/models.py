"""Collaborative models, implemented directly rather than pulled from a library.

Two reasons this is hand-written:

1. The assignment asks for a system built from scratch, and ALS is the one
   component where the arithmetic is worth showing.
2. `implicit` ships no wheel for Python 3.13, so the alternative was pinning an
   older interpreter for a ~60-line algorithm.

Both models are small enough to fit comfortably in memory at this scale
(610 users x 9,742 items), so everything below is dense-friendly.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse


def item_knn(
    normalised: sparse.csc_matrix, top_k: int = 50, block: int = 512
) -> tuple[np.ndarray, np.ndarray]:
    """Top-k cosine neighbours per item.

    `normalised` must have unit-norm columns, so `Sᵀ S` is already a cosine
    similarity matrix. The full 9,742² matrix is ~380 MB dense; we never
    materialise it, computing it in column blocks and keeping only the top k.

    Returns (indices, similarities), both shape (n_items, top_k), row-aligned to
    item index. Self-similarity is excluded. Rows for items with no ratings are
    filled with -1 / 0.0 — the caller must treat them as "no CF signal".
    """
    n_items = normalised.shape[1]
    csr = normalised.T.tocsr()  # items x users

    idx_out = np.full((n_items, top_k), -1, dtype=np.int32)
    sim_out = np.zeros((n_items, top_k), dtype=np.float32)

    for start in range(0, n_items, block):
        stop = min(start + block, n_items)
        # (block x users) @ (users x items) -> (block x items) cosine scores
        sims = np.asarray((csr[start:stop] @ normalised).todense(), dtype=np.float32)

        # Exclude self-similarity.
        for local, item in enumerate(range(start, stop)):
            sims[local, item] = -np.inf

        # argpartition is O(n) vs a full sort's O(n log n); we only need the top k.
        k = min(top_k, sims.shape[1] - 1)
        part = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        part_scores = np.take_along_axis(sims, part, axis=1)
        order = np.argsort(-part_scores, axis=1)
        neighbours = np.take_along_axis(part, order, axis=1)
        scores = np.take_along_axis(part_scores, order, axis=1)

        # Similarity of exactly 0 means "no co-rating evidence", not a weak
        # neighbour. Mark those slots empty so scoring can skip them.
        scores[~np.isfinite(scores)] = 0.0
        neighbours = np.where(scores > 0, neighbours, -1)
        scores = np.where(scores > 0, scores, 0.0)

        idx_out[start:stop, :k] = neighbours.astype(np.int32)
        sim_out[start:stop, :k] = scores.astype(np.float32)

    return idx_out, sim_out


def _solve_side(
    fixed: np.ndarray,
    confidence: sparse.csr_matrix,
    regularisation: float,
) -> np.ndarray:
    """One ALS half-step (Hu, Koren & Volinsky 2008).

    For each row u we solve

        (YᵀY + Yᵀ(C_u − I)Y + λI) x_u = Yᵀ C_u p(u)

    The YᵀY term is shared across all rows and computed once. Because C_u − I is
    zero everywhere the user has no interaction, the per-user correction only
    touches that user's observed items — which is what makes this tractable.
    """
    n_factors = fixed.shape[1]
    gramian = fixed.T @ fixed  # shared YᵀY
    lambda_eye = regularisation * np.eye(n_factors, dtype=np.float64)
    out = np.zeros((confidence.shape[0], n_factors), dtype=np.float64)

    indptr, indices, data = confidence.indptr, confidence.indices, confidence.data

    for row in range(confidence.shape[0]):
        start, stop = indptr[row], indptr[row + 1]
        if start == stop:
            continue  # no interactions: leave factors at zero
        cols = indices[start:stop]
        conf = data[start:stop].astype(np.float64)

        sub = fixed[cols]  # (n_observed x factors)
        # Yᵀ(C_u − I)Y  — only observed items contribute.
        correction = sub.T @ (sub * (conf - 1.0)[:, None])
        # Preference p(u) is 1 wherever an interaction exists, so Yᵀ C_u p(u)
        # collapses to a confidence-weighted sum of the observed item factors.
        target = sub.T @ conf

        out[row] = np.linalg.solve(gramian + correction + lambda_eye, target)

    return out


def als(
    confidence: sparse.csr_matrix,
    n_factors: int = 64,
    regularisation: float = 0.05,
    iterations: int = 15,
    seed: int = 42,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Alternating Least Squares for implicit feedback.

    Returns (user_factors, item_factors) as float32, shapes (n_users, f) and
    (n_items, f). Score a pair with a plain dot product.
    """
    rng = np.random.default_rng(seed)
    n_users, n_items = confidence.shape

    # Small random init; scale keeps the first solve well-conditioned.
    user_factors = rng.normal(0, 0.01, (n_users, n_factors))
    item_factors = rng.normal(0, 0.01, (n_items, n_factors))

    confidence_csr = confidence.tocsr()
    confidence_csc = confidence.T.tocsr()  # items x users

    for iteration in range(iterations):
        user_factors = _solve_side(item_factors, confidence_csr, regularisation)
        item_factors = _solve_side(user_factors, confidence_csc, regularisation)

        if verbose:
            loss = _confidence_loss(confidence_csr, user_factors, item_factors, regularisation)
            print(f"    iter {iteration + 1:>2}/{iterations}  loss {loss:,.1f}")

    return user_factors.astype(np.float32), item_factors.astype(np.float32)


def _confidence_loss(
    confidence: sparse.csr_matrix,
    user_factors: np.ndarray,
    item_factors: np.ndarray,
    regularisation: float,
) -> float:
    """Observed-entry reconstruction error plus L2 penalty.

    This is the *observed* part of the HKV objective only. The unobserved term
    (every zero cell pulled toward 0 with confidence 1) is omitted because
    evaluating it costs a dense n_users x n_items pass and we only need this
    number to confirm the iteration is converging, not to report it.
    """
    total = 0.0
    indptr, indices, data = confidence.indptr, confidence.indices, confidence.data
    for row in range(confidence.shape[0]):
        start, stop = indptr[row], indptr[row + 1]
        if start == stop:
            continue
        cols = indices[start:stop]
        predicted = item_factors[cols] @ user_factors[row]
        total += float(np.sum(data[start:stop] * (1.0 - predicted) ** 2))
    penalty = regularisation * (float(np.sum(user_factors**2)) + float(np.sum(item_factors**2)))
    return total + penalty
