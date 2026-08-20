"""Offline evaluation: baselines, ablations, and hyperparameter tuning.

    python scripts/evaluate.py --all          # full table -> docs/EVALUATION.md
    python scripts/evaluate.py --tune         # grid search on the validation split
    python scripts/evaluate.py --models hybrid_mmr

**Leakage control.** The committed artifacts were trained on every rating, so
using them here would let each model see the test period it is being scored on.
This script therefore *retrains* ALS, item-kNN and popularity on the training
split alone. Content embeddings are reused as-is: they are derived from film
metadata, never from ratings, so they cannot leak interaction data.

Hyperparameters are chosen on a validation slice carved from the end of
training, never on the test split we report.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import metrics as M  # noqa: E402
from content import popularity_table  # noqa: E402
from matrix import (  # noqa: E402
    ARTIFACT_DIR,
    REPO_ROOT,
    build_confidence_matrix,
    build_id_maps,
    build_rating_matrix,
    l2_normalise_columns,
    load_frames,
    mean_center_rows,
)
from models import als, item_knn  # noqa: E402
from split import Split, cohort_of, temporal_split  # noqa: E402

from api.recommender import Components, Recommender  # noqa: E402

K = 10
SEED = 42


@dataclass
class Result:
    name: str
    precision: float
    recall: float
    ndcg: float
    map_: float
    coverage: float
    novelty: float
    diversity: float
    gini: float
    p50_ms: float
    p95_ms: float
    n_users: int

    def row(self) -> str:
        return (
            f"| {self.name} | {self.precision:.4f} | {self.recall:.4f} | {self.ndcg:.4f} | "
            f"{self.map_:.4f} | {self.coverage:.1f}% | {self.novelty:.2f} | "
            f"{self.diversity:.3f} | {self.gini:.3f} | {self.p50_ms:.1f} | {self.p95_ms:.1f} |"
        )


def fit_components(
    train: sparse.csr_matrix,
    embeddings: np.ndarray,
    catalogue: pd.DataFrame,
    ratings: pd.DataFrame,
    item_ids: np.ndarray,
    factors: int,
    regularisation: float,
    iterations: int,
    alpha: float,
    verbose: bool = False,
) -> Components:
    """Refit every rating-derived model on the training split only."""
    normalised = l2_normalise_columns(mean_center_rows(train))
    knn_idx, knn_sim = item_knn(normalised, top_k=50)

    confidence = build_confidence_matrix(train, alpha=alpha)
    user_factors, item_factors = als(
        confidence,
        n_factors=factors,
        regularisation=regularisation,
        iterations=iterations,
        seed=SEED,
        verbose=verbose,
    )

    # Popularity must also come from training rows only, or the popularity
    # baseline gets a preview of which films become popular in the test period.
    train_coo = train.tocoo()
    train_ratings = pd.DataFrame(
        {
            "movieId": item_ids[train_coo.col],
            "rating": train_coo.data,
        }
    )
    table = popularity_table(train_ratings, item_ids)

    return Components(
        user_factors=user_factors,
        item_factors=item_factors,
        knn_indices=knn_idx,
        knn_similarities=knn_sim,
        embeddings=embeddings,
        popularity=table["popularity"].to_numpy(dtype=np.float32),
        novelty=table["novelty"].to_numpy(dtype=np.float32),
        interactions=train,
        catalogue=catalogue,
    )


# ------------------------------------------------------------------ rankers


def rank_random(rng: np.random.Generator, n_items: int, seen: set[int], k: int) -> np.ndarray:
    pool = np.setdiff1d(np.arange(n_items), np.fromiter(seen, dtype=np.int64), assume_unique=False)
    return rng.choice(pool, size=min(k, len(pool)), replace=False)


def rank_popularity(popularity: np.ndarray, seen: set[int], k: int) -> np.ndarray:
    order = np.argsort(-popularity)
    return np.array([i for i in order if int(i) not in seen][:k], dtype=np.int64)


def evaluate_model(
    name: str,
    ranker,
    split: Split,
    components: Components,
    k: int = K,
) -> tuple[Result, dict[int, np.ndarray]]:
    """Run a ranker over every evaluable user and aggregate."""
    precisions, recalls, ndcgs, maps_, diversities, novelties = [], [], [], [], [], []
    latencies: list[float] = []
    all_ranked: list[np.ndarray] = []
    per_user: dict[int, np.ndarray] = {}

    for user, relevant_items in split.truth.items():
        relevant = {int(i) for i in relevant_items}

        start = time.perf_counter()
        ranked = ranker(user)
        latencies.append((time.perf_counter() - start) * 1000)

        ranked = np.asarray(ranked, dtype=np.int64)
        per_user[user] = ranked
        all_ranked.append(ranked)

        precisions.append(M.precision_at_k(ranked, relevant, k))
        recalls.append(M.recall_at_k(ranked, relevant, k))
        ndcgs.append(M.ndcg_at_k(ranked, relevant, k))
        maps_.append(M.average_precision_at_k(ranked, relevant, k))
        diversities.append(M.intra_list_diversity(ranked, components.embeddings, k))
        novelties.append(M.novelty(ranked, components.novelty, k))

    return (
        Result(
            name=name,
            precision=float(np.mean(precisions)),
            recall=float(np.mean(recalls)),
            ndcg=float(np.mean(ndcgs)),
            map_=float(np.mean(maps_)),
            coverage=M.catalogue_coverage(all_ranked, components.n_items, k),
            novelty=float(np.mean(novelties)),
            diversity=float(np.mean(diversities)),
            gini=M.gini(all_ranked, components.n_items, k),
            p50_ms=float(np.percentile(latencies, 50)),
            p95_ms=float(np.percentile(latencies, 95)),
            n_users=len(split.truth),
        ),
        per_user,
    )


def build_rankers(components: Components, weights: dict[str, float], lambda_: float) -> dict:
    """Every model in the comparison table, as user -> ranked items callables."""
    rng = np.random.default_rng(SEED)
    recommender = Recommender(components, weights=weights, mmr_lambda=lambda_)

    def seen_of(user: int) -> set[int]:
        return {int(i) for i in components.interactions[user].indices}

    def only(signal: str):
        single = {"als": 0.0, "item_cf": 0.0, "content": 0.0, "popularity": 0.0}
        single[signal] = 1.0

        def ranker(user: int) -> np.ndarray:
            recs = recommender.recommend(user, k=K, weights=single, mmr_lambda=1.0)
            return np.array([r.item for r in recs], dtype=np.int64)

        return ranker

    def hybrid(mmr: float):
        def ranker(user: int) -> np.ndarray:
            recs = recommender.recommend(user, k=K, weights=weights, mmr_lambda=mmr)
            return np.array([r.item for r in recs], dtype=np.int64)

        return ranker

    return {
        "Random": lambda u: rank_random(rng, components.n_items, seen_of(u), K),
        "Popularity": lambda u: rank_popularity(components.popularity, seen_of(u), K),
        "Item-kNN": only("item_cf"),
        "ALS": only("als"),
        "Content-only": only("content"),
        "Hybrid": hybrid(1.0),
        "Hybrid + MMR (shipped)": hybrid(lambda_),
    }


def tune(
    validation: Split,
    components: Components,
    output: Path,
) -> tuple[dict[str, float], float]:
    """Grid search the blend weights and lambda on the validation split."""
    grids = [
        {"als": a, "item_cf": c, "content": t, "popularity": p}
        for a in (0.2, 0.3, 0.4, 0.5, 0.6)
        for c in (0.1, 0.2, 0.3, 0.4)
        for t in (0.05, 0.1, 0.2, 0.3)
        for p in (0.0, 0.05, 0.1)
        if abs(a + c + t + p - 1.0) < 1e-9
    ]
    print(f"grid: {len(grids)} weight combinations")

    best_weights, best_ndcg = None, -1.0
    for n, weights in enumerate(grids, start=1):
        recommender = Recommender(components, weights=weights, mmr_lambda=1.0)
        scores = []
        for user, relevant_items in validation.truth.items():
            recs = recommender.recommend(user, k=K, mmr_lambda=1.0)
            ranked = np.array([r.item for r in recs], dtype=np.int64)
            scores.append(M.ndcg_at_k(ranked, {int(i) for i in relevant_items}, K))
        ndcg = float(np.mean(scores))
        if ndcg > best_ndcg:
            best_ndcg, best_weights = ndcg, weights
            print(f"  [{n}/{len(grids)}] new best NDCG@{K} {ndcg:.4f}  {weights}")

    print(f"\nbest weights {best_weights} -> validation NDCG@{K} {best_ndcg:.4f}")

    print("\ntuning lambda (diversity/accuracy tradeoff)")
    lambda_rows = []
    for lambda_ in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5):
        recommender = Recommender(components, weights=best_weights, mmr_lambda=lambda_)
        ndcgs, diversities = [], []
        for user, relevant_items in validation.truth.items():
            recs = recommender.recommend(user, k=K)
            ranked = np.array([r.item for r in recs], dtype=np.int64)
            ndcgs.append(M.ndcg_at_k(ranked, {int(i) for i in relevant_items}, K))
            diversities.append(M.intra_list_diversity(ranked, components.embeddings, K))
        row = {
            "lambda": lambda_,
            "ndcg": float(np.mean(ndcgs)),
            "diversity": float(np.mean(diversities)),
        }
        lambda_rows.append(row)
        print(f"  lambda={lambda_:.1f}  NDCG {row['ndcg']:.4f}  ILD {row['diversity']:.3f}")

    # Pick the smallest lambda whose NDCG stays within 5% of the best: buy the
    # most diversity available for a cost we are willing to state out loud.
    ceiling = max(r["ndcg"] for r in lambda_rows)
    affordable = [r for r in lambda_rows if r["ndcg"] >= 0.95 * ceiling]
    best_lambda = min(r["lambda"] for r in affordable)
    print(f"\nselected lambda = {best_lambda} (within 5% of best NDCG, maximum diversity)")

    output.write_text(
        json.dumps(
            {
                "weights": best_weights,
                "lambda": best_lambda,
                "validation_ndcg": best_ndcg,
                "lambda_sweep": lambda_rows,
                "n_validation_users": validation.n_evaluable,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return best_weights, best_lambda


def write_report(
    results: list[Result],
    cohorts: dict[str, dict[str, float]],
    config: dict,
    path: Path,
) -> None:
    lines = [
        "# Evaluation Results",
        "",
        "Generated by `python scripts/evaluate.py --all`. Do not edit by hand.",
        "",
        f"- Protocol: per-user temporal split, earliest 80% train / latest 20% test, k = {K}",
        "- Relevance: a held-out item counts as relevant if the user rated it >= 4.0",
        f"- Evaluable users: {results[0].n_users} of 610",
        "- Every rating-derived model (ALS, item-kNN, popularity) is refit on the",
        "  training split only. Content embeddings come from film metadata and",
        "  carry no interaction signal, so they are reused as built.",
        "",
        f"Configuration: `{json.dumps(config)}`",
        "",
        "## Model comparison",
        "",
        "| Model | P@10 | R@10 | NDCG@10 | MAP@10 | Coverage | Novelty | ILD | Gini | p50 (ms) | p95 (ms) |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    lines.extend(r.row() for r in results)
    lines += [
        "",
        "**Gini** measures how evenly exposure is spread across items (0 = uniform,",
        "1 = all impressions on one film). Coverage counts how many films are reached;",
        "Gini catches the case where a system reaches many films but still sends",
        "almost every impression to a handful.",
        "",
        "## Per-cohort NDCG@10 (shipped model)",
        "",
        "Cohorts are training-history length. MovieLens guarantees at least 20 ratings",
        "per user, so 'light' means light *for this dataset* — there are no genuinely",
        "cold users to measure.",
        "",
        "| Cohort | Users | NDCG@10 |",
        "|---|---|---|",
    ]
    for name, stats in cohorts.items():
        lines.append(f"| {name} | {int(stats['n'])} | {stats['ndcg']:.4f} |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="evaluate every model")
    parser.add_argument("--tune", action="store_true", help="grid search on validation")
    parser.add_argument("--factors", type=int, default=64)
    parser.add_argument("--regularisation", type=float, default=0.05)
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--alpha", type=float, default=40.0)
    args = parser.parse_args()

    frames = load_frames()
    maps = build_id_maps(frames["ratings"], frames["movies"])
    embeddings = np.load(ARTIFACT_DIR / "content_emb.npy")
    catalogue = pd.read_parquet(ARTIFACT_DIR / "catalogue.parquet")

    print("building temporal split")
    test_split, validation_split = temporal_split(
        frames["ratings"], maps, test_fraction=0.2, validation_fraction=0.1
    )
    full = build_rating_matrix(frames["ratings"], maps)
    print(f"  full {full.nnz:,} -> train {test_split.train.nnz:,}")
    print(f"  evaluable users: test {test_split.n_evaluable}, val {validation_split.n_evaluable}")

    fit_kwargs = dict(
        embeddings=embeddings,
        catalogue=catalogue,
        ratings=frames["ratings"],
        item_ids=maps.item_ids,
        factors=args.factors,
        regularisation=args.regularisation,
        iterations=args.iterations,
        alpha=args.alpha,
    )

    weights_path = ARTIFACT_DIR / "tuned_weights.json"
    weights, lambda_ = None, None

    if args.tune:
        print("\nrefitting on validation-training split")
        validation_components = fit_components(validation_split.train, **fit_kwargs)
        weights, lambda_ = tune(validation_split, validation_components, weights_path)

    if weights is None:
        if weights_path.exists():
            payload = json.loads(weights_path.read_text(encoding="utf-8"))
            weights, lambda_ = payload["weights"], payload["lambda"]
            print(f"\nusing tuned config from {weights_path.name}: {weights}, lambda={lambda_}")
        else:
            from api.recommender import DEFAULT_LAMBDA, DEFAULT_WEIGHTS

            weights, lambda_ = DEFAULT_WEIGHTS, DEFAULT_LAMBDA
            print(f"\nno tuned config; using defaults {weights}, lambda={lambda_}")

    if not args.all:
        return 0

    print("\nrefitting on training split for test evaluation")
    components = fit_components(test_split.train, verbose=True, **fit_kwargs)

    results = []
    per_user_shipped: dict[int, np.ndarray] = {}
    for name, ranker in build_rankers(components, weights, lambda_).items():
        print(f"  evaluating {name}")
        result, per_user = evaluate_model(name, ranker, test_split, components)
        results.append(result)
        if name.startswith("Hybrid + MMR"):
            per_user_shipped = per_user

    cohorts: dict[str, dict[str, float]] = {}
    for user, ranked in per_user_shipped.items():
        cohort = cohort_of(test_split.train, user)
        entry = cohorts.setdefault(cohort, {"n": 0.0, "ndcg": 0.0})
        entry["n"] += 1
        entry["ndcg"] += M.ndcg_at_k(ranked, {int(i) for i in test_split.truth[user]}, K)
    for entry in cohorts.values():
        entry["ndcg"] /= max(entry["n"], 1)
    cohorts = {k: cohorts[k] for k in ("light", "medium", "heavy") if k in cohorts}

    print()
    for result in results:
        print(f"  {result.name:<24} NDCG@10 {result.ndcg:.4f}  ILD {result.diversity:.3f}")

    config = {
        "weights": weights,
        "lambda": lambda_,
        "factors": args.factors,
        "regularisation": args.regularisation,
        "iterations": args.iterations,
        "alpha": args.alpha,
    }
    docs = REPO_ROOT / "docs"
    docs.mkdir(exist_ok=True)
    write_report(results, cohorts, config, docs / "EVALUATION.md")
    (ARTIFACT_DIR / "evaluation.json").write_text(
        json.dumps(
            {"config": config, "results": [r.__dict__ for r in results], "cohorts": cohorts},
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
