"""Scoring, reranking, and explanation — the whole recommendation decision.

This module is deliberately the *only* implementation of the ranking logic. The
offline evaluation imports it too, so the numbers in the README describe the
code that actually serves requests rather than a parallel reimplementation that
might have drifted from it.

Nothing here trains. Every expensive quantity was precomputed by
`scripts/build_artifacts.py`; this is matrix slices, blending, and bookkeeping.

Pipeline, per README section 3:

    1. candidate generation   ~300 items from kNN + ALS + content
    2. scoring                weighted blend of four normalised signals
    3. reranking              MMR over content embeddings for diversity
    4. explanation            per-component contributions + strongest seed
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

DEFAULT_WEIGHTS = {"als": 0.40, "item_cf": 0.30, "content": 0.20, "popularity": 0.10}
DEFAULT_LAMBDA = 0.75
DEFAULT_CANDIDATES = 300

# Below this many rated items we stop claiming the result is personalised.
COLD_START_MINIMUM = 3


@dataclass
class Components:
    """Everything the scorer needs, all index-aligned on item index."""

    user_factors: np.ndarray  # (n_users, f)
    item_factors: np.ndarray  # (n_items, f)
    knn_indices: np.ndarray  # (n_items, k) int32, -1 padded
    knn_similarities: np.ndarray  # (n_items, k) float32
    embeddings: np.ndarray  # (n_items, d) L2-normalised
    popularity: np.ndarray  # (n_items,) in [0, 1]
    novelty: np.ndarray  # (n_items,) self-information
    interactions: sparse.csr_matrix  # (n_users, n_items) observed ratings
    catalogue: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def n_items(self) -> int:
        return self.item_factors.shape[0]

    @classmethod
    def load(cls, artifact_dir: Path, interactions: sparse.csr_matrix) -> Components:
        neighbours = np.load(artifact_dir / "neighbors_top50.npz")
        popularity_frame = pd.read_parquet(artifact_dir / "popularity.parquet")
        catalogue = pd.read_parquet(artifact_dir / "catalogue.parquet")
        return cls(
            user_factors=np.load(artifact_dir / "user_factors.npy"),
            item_factors=np.load(artifact_dir / "item_factors.npy"),
            knn_indices=neighbours["indices"],
            knn_similarities=neighbours["similarities"],
            embeddings=np.load(artifact_dir / "content_emb.npy"),
            popularity=popularity_frame["popularity"].to_numpy(dtype=np.float32),
            novelty=popularity_frame["novelty"].to_numpy(dtype=np.float32),
            interactions=interactions,
            catalogue=catalogue,
        )


@dataclass
class Recommendation:
    item: int
    score: float
    contributions: dict[str, float]
    seed_item: int | None
    seed_similarity: float


def _minmax(values: np.ndarray) -> np.ndarray:
    """Scale to [0, 1] within the candidate set.

    Each signal lives on its own scale — ALS dot products are unbounded, cosine
    is [-1, 1], popularity is already [0, 1]. Without this, the weights would
    encode scale rather than importance.

    A constant vector maps to zeros, not ones: if a signal cannot discriminate
    between these candidates it should contribute nothing, not a flat bonus.
    """
    if values.size == 0:
        return values
    low, high = float(values.min()), float(values.max())
    if high - low < 1e-12:
        return np.zeros_like(values)
    return (values - low) / (high - low)


class Recommender:
    """Hybrid scorer over precomputed artifacts."""

    def __init__(
        self,
        components: Components,
        weights: dict[str, float] | None = None,
        mmr_lambda: float = DEFAULT_LAMBDA,
        n_candidates: int = DEFAULT_CANDIDATES,
    ) -> None:
        self.c = components
        self.weights = dict(weights or DEFAULT_WEIGHTS)
        self.mmr_lambda = mmr_lambda
        self.n_candidates = n_candidates

    # ---------------------------------------------------------------- history

    def history(self, user: int) -> tuple[np.ndarray, np.ndarray]:
        """(item indices, ratings) this user has rated."""
        row = self.c.interactions[user]
        return row.indices, row.data

    def liked(self, user: int, threshold: float = 4.0) -> np.ndarray:
        items, ratings = self.history(user)
        return items[ratings >= threshold]

    # ------------------------------------------------------ candidate stage

    def _candidates(self, user: int, seeds: np.ndarray, exclude: set[int]) -> np.ndarray:
        """Recall-oriented retrieval. Scoring all 9,742 items per request is
        wasteful; this narrows to a few hundred plausible ones."""
        pool: set[int] = set()

        # 1. Collaborative neighbours of everything the user liked.
        if seeds.size:
            neighbours = self.c.knn_indices[seeds].ravel()
            pool.update(int(i) for i in neighbours if i >= 0)

        # 2. Top ALS items, if this user has latent factors.
        if user >= 0 and user < self.c.user_factors.shape[0]:
            als_scores = self.c.item_factors @ self.c.user_factors[user]
            pool.update(int(i) for i in np.argpartition(-als_scores, 200)[:200])

        # 3. Content neighbours of the user's profile.
        profile = self._content_profile(seeds)
        if profile is not None:
            content_scores = self.c.embeddings @ profile
            pool.update(int(i) for i in np.argpartition(-content_scores, 200)[:200])

        # 4. Popularity backfill guarantees a full slate for sparse users.
        if len(pool) < self.n_candidates:
            top_popular = np.argsort(-self.c.popularity)[: self.n_candidates * 2]
            pool.update(int(i) for i in top_popular)

        candidates = np.array(sorted(pool - exclude), dtype=np.int64)
        return candidates

    def _content_profile(self, seeds: np.ndarray) -> np.ndarray | None:
        """Mean of the seed embeddings, renormalised to the unit sphere."""
        if seeds.size == 0:
            return None
        profile = self.c.embeddings[seeds].mean(axis=0)
        norm = float(np.linalg.norm(profile))
        return profile / norm if norm > 0 else None

    # --------------------------------------------------------- scoring stage

    def _raw_signals(
        self, user: int, seeds: np.ndarray, candidates: np.ndarray
    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
        """Un-normalised component scores, plus the best seed per candidate.

        The seed bookkeeping happens here rather than in a separate explanation
        pass, so the item named in "Because you watched X" is by construction
        the one that actually drove the score.
        """
        n = len(candidates)

        if 0 <= user < self.c.user_factors.shape[0]:
            als = self.c.item_factors[candidates] @ self.c.user_factors[user]
        else:
            als = np.zeros(n, dtype=np.float32)

        # Item-CF: sum of similarities from the user's liked items, accumulated
        # over the truncated neighbour lists. Tracks which seed contributed most.
        item_cf = np.zeros(n, dtype=np.float32)
        best_seed = np.full(n, -1, dtype=np.int64)
        best_seed_similarity = np.zeros(n, dtype=np.float32)

        if seeds.size:
            position = {int(item): idx for idx, item in enumerate(candidates)}
            for seed in seeds:
                neighbours = self.c.knn_indices[seed]
                similarities = self.c.knn_similarities[seed]
                for neighbour, similarity in zip(neighbours, similarities, strict=True):
                    if neighbour < 0:
                        continue
                    idx = position.get(int(neighbour))
                    if idx is None:
                        continue
                    item_cf[idx] += similarity
                    if similarity > best_seed_similarity[idx]:
                        best_seed_similarity[idx] = similarity
                        best_seed[idx] = int(seed)

        profile = self._content_profile(seeds)
        content = (
            self.c.embeddings[candidates] @ profile
            if profile is not None
            else np.zeros(n, dtype=np.float32)
        )

        signals = {
            "als": als.astype(np.float32),
            "item_cf": item_cf,
            "content": content.astype(np.float32),
            "popularity": self.c.popularity[candidates],
        }
        return signals, best_seed, best_seed_similarity

    # ---------------------------------------------------------- rerank stage

    def _mmr(
        self, candidates: np.ndarray, scores: np.ndarray, k: int, lambda_: float
    ) -> list[int]:
        """Maximal Marginal Relevance selection.

        Greedily picks the item maximising

            lambda * relevance - (1 - lambda) * max_similarity_to_already_chosen

        Trades a little ranking accuracy for a large gain in slate variety. With
        lambda = 1 this degenerates to plain relevance ordering, which is how
        the ablation in the evaluation isolates its effect.
        """
        if lambda_ >= 1.0:
            return list(np.argsort(-scores)[:k])

        vectors = self.c.embeddings[candidates]
        selected: list[int] = []
        remaining = set(range(len(candidates)))
        # Similarity to the selected set, maintained incrementally.
        max_similarity = np.zeros(len(candidates), dtype=np.float32)

        for _ in range(min(k, len(candidates))):
            best_idx, best_value = -1, -np.inf
            for idx in remaining:
                value = lambda_ * scores[idx] - (1.0 - lambda_) * max_similarity[idx]
                if value > best_value:
                    best_value, best_idx = value, idx
            if best_idx < 0:
                break
            selected.append(best_idx)
            remaining.discard(best_idx)
            if remaining:
                similarity_to_new = vectors @ vectors[best_idx]
                max_similarity = np.maximum(max_similarity, similarity_to_new)

        return selected

    # ------------------------------------------------------------- public API

    def recommend(
        self,
        user: int,
        k: int = 10,
        seeds: np.ndarray | None = None,
        exclude_seen: bool = True,
        mmr_lambda: float | None = None,
        weights: dict[str, float] | None = None,
    ) -> list[Recommendation]:
        """Ranked recommendations with per-component explanations.

        `seeds` overrides the user's history, which is how cold start works: a
        caller supplies chosen films for a user that has no row in the matrix
        (pass user = -1).
        """
        weights = weights or self.weights
        lambda_ = self.mmr_lambda if mmr_lambda is None else mmr_lambda

        if seeds is None:
            seeds = self.liked(user) if user >= 0 else np.array([], dtype=np.int64)

        exclude: set[int] = set()
        if exclude_seen and user >= 0:
            exclude.update(int(i) for i in self.history(user)[0])
        exclude.update(int(i) for i in seeds)

        candidates = self._candidates(user, seeds, exclude)
        if candidates.size == 0:
            return []

        signals, best_seed, best_seed_similarity = self._raw_signals(user, seeds, candidates)

        # Normalise within the candidate set, then blend. Keeping the weighted
        # parts is what makes the explanation exact rather than reconstructed.
        parts = {name: weights.get(name, 0.0) * _minmax(values) for name, values in signals.items()}
        total = np.sum(list(parts.values()), axis=0)

        order = self._mmr(candidates, total, k, lambda_)

        results: list[Recommendation] = []
        for idx in order:
            share = float(total[idx])
            contributions = (
                {name: float(part[idx]) / share for name, part in parts.items()}
                if share > 1e-12
                else dict.fromkeys(parts, 0.0)
            )
            seed = int(best_seed[idx])
            results.append(
                Recommendation(
                    item=int(candidates[idx]),
                    score=share,
                    contributions=contributions,
                    seed_item=seed if seed >= 0 else None,
                    seed_similarity=float(best_seed_similarity[idx]),
                )
            )
        return results

    def similar(self, item: int, k: int = 10) -> list[tuple[int, float]]:
        """Item-to-item neighbours, falling back to content for cold items."""
        neighbours = self.c.knn_indices[item]
        similarities = self.c.knn_similarities[item]
        pairs = [(int(i), float(s)) for i, s in zip(neighbours, similarities, strict=True) if i >= 0]
        if len(pairs) >= k:
            return pairs[:k]

        # Cold item: no collaborative signal at all (20 such films exist).
        scores = self.c.embeddings @ self.c.embeddings[item]
        scores[item] = -np.inf
        seen = {i for i, _ in pairs}
        for idx in np.argsort(-scores):
            if len(pairs) >= k:
                break
            if int(idx) not in seen:
                pairs.append((int(idx), float(scores[idx])))
        return pairs[:k]

    def search(self, query_vector: np.ndarray, k: int = 10) -> list[tuple[int, float]]:
        """Semantic search over content embeddings."""
        norm = float(np.linalg.norm(query_vector))
        if norm == 0:
            return []
        scores = self.c.embeddings @ (query_vector / norm)
        top = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in top]
