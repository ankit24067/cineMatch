"""Build every offline artifact the serving layer needs.

    python scripts/build_artifacts.py

Everything expensive happens here, exactly once. The API then does lookups and
matrix slices only — it never trains, which is what keeps p95 latency in the
tens of milliseconds on free-tier hardware (README section 14).

Outputs, all written to `artifacts/`:

    id_maps.json          MovieLens id <-> contiguous index, the shared contract
    neighbors_top50.npz   item-kNN indices + similarities
    user_factors.npy      ALS user latent factors
    item_factors.npy      ALS item latent factors
    content_emb.npy       MiniLM document embeddings, L2-normalised
    popularity.parquet    counts, damped rating, novelty, popularity term
    catalogue.parquet     titles, year, genres, poster URL for the UI
    manifest.json         hyperparameters and shapes, for reproducibility

Run `python scripts/fetch_tmdb.py` first for overviews and posters; the build
works without it but the content model is markedly weaker.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

import content as content_mod
from matrix import (
    ARTIFACT_DIR,
    POSITIVE_THRESHOLD,
    build_confidence_matrix,
    build_id_maps,
    build_rating_matrix,
    l2_normalise_columns,
    load_frames,
    mean_center_rows,
)
from models import als, item_knn

TMDB_CACHE = ARTIFACT_DIR / "tmdb.json"


def _step(label: str) -> float:
    print(f"\n[{label}]")
    return time.perf_counter()


def _done(started: float) -> float:
    elapsed = time.perf_counter() - started
    print(f"  took {elapsed:.1f}s")
    return elapsed


def build_catalogue(
    movies: pd.DataFrame, item_ids: np.ndarray, tmdb: dict[str, dict]
) -> pd.DataFrame:
    """Item metadata the frontend renders. Index-aligned with every artifact."""
    indexed = movies.set_index("movieId").loc[item_ids]
    titles = indexed["title"].astype(str).to_numpy()

    years, clean_titles = [], []
    for title in titles:
        name, year = content_mod._split_title_year(title)
        clean_titles.append(name)
        years.append(int(year) if year else 0)

    records = [tmdb.get(str(int(m)), {}) for m in item_ids]
    return pd.DataFrame(
        {
            "index": np.arange(len(item_ids), dtype=np.int32),
            "movie_id": item_ids.astype(np.int64),
            "title": clean_titles,
            "year": np.asarray(years, dtype=np.int16),
            "genres": indexed["genres"].astype(str).to_numpy(),
            "poster": [r.get("poster", "") for r in records],
            "overview": [r.get("overview", "")[:400] for r in records],
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factors", type=int, default=64)
    parser.add_argument("--regularisation", type=float, default=0.05)
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--alpha", type=float, default=40.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--skip-embeddings", action="store_true", help="reuse content_emb.npy")
    args = parser.parse_args()

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    wall = time.perf_counter()

    started = _step("load")
    frames = load_frames()
    maps = build_id_maps(frames["ratings"], frames["movies"])
    print(f"  {maps.n_users} users x {maps.n_items} items")
    tmdb: dict[str, dict] = {}
    if TMDB_CACHE.exists():
        tmdb = json.loads(TMDB_CACHE.read_text(encoding="utf-8"))
        print(f"  TMDB cache: {len(tmdb)} films")
    else:
        print("  ! no TMDB cache — content documents will be genres/tags only")
    maps.save(ARTIFACT_DIR / "id_maps.json")
    _done(started)

    started = _step("rating matrices")
    ratings = build_rating_matrix(frames["ratings"], maps)
    confidence = build_confidence_matrix(ratings, alpha=args.alpha)
    normalised = l2_normalise_columns(mean_center_rows(ratings))
    print(f"  {ratings.nnz:,} ratings, {confidence.nnz:,} positives (>= {POSITIVE_THRESHOLD})")
    _done(started)

    started = _step(f"item-kNN (top {args.top_k})")
    neighbour_idx, neighbour_sim = item_knn(normalised, top_k=args.top_k)
    reachable = int((neighbour_idx[:, 0] >= 0).sum())
    np.savez_compressed(
        ARTIFACT_DIR / "neighbors_top50.npz", indices=neighbour_idx, similarities=neighbour_sim
    )
    print(f"  {reachable}/{maps.n_items} items have collaborative neighbours")
    _done(started)

    started = _step(f"ALS (f={args.factors}, reg={args.regularisation}, iters={args.iterations})")
    user_factors, item_factors = als(
        confidence,
        n_factors=args.factors,
        regularisation=args.regularisation,
        iterations=args.iterations,
    )
    np.save(ARTIFACT_DIR / "user_factors.npy", user_factors)
    np.save(ARTIFACT_DIR / "item_factors.npy", item_factors)
    _done(started)

    embedding_path = ARTIFACT_DIR / "content_emb.npy"
    coverage: dict[str, float] = {}
    if args.skip_embeddings and embedding_path.exists():
        print("\n[content embeddings] skipped, reusing existing file")
        embeddings = np.load(embedding_path)
    else:
        started = _step("content embeddings (MiniLM)")
        documents, coverage = content_mod.build_documents(
            frames["movies"], frames["tags"], maps.item_ids, tmdb
        )
        print(f"  {coverage}")
        embeddings = content_mod.encode(documents)
        np.save(embedding_path, embeddings)
        (ARTIFACT_DIR / "documents_sample.json").write_text(
            json.dumps(documents[:5], indent=2, ensure_ascii=False), encoding="utf-8"
        )
        _done(started)

    started = _step("popularity + catalogue")
    popularity = content_mod.popularity_table(frames["ratings"], maps.item_ids)
    popularity.to_parquet(ARTIFACT_DIR / "popularity.parquet", index=False)
    catalogue = build_catalogue(frames["movies"], maps.item_ids, tmdb)
    catalogue.to_parquet(ARTIFACT_DIR / "catalogue.parquet", index=False)
    with_poster = int((catalogue["poster"] != "").sum())
    print(f"  posters available for {with_poster}/{len(catalogue)} films")
    _done(started)

    manifest = {
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "dataset": "ml-latest-small",
        "n_users": maps.n_users,
        "n_items": maps.n_items,
        "n_ratings": int(ratings.nnz),
        "n_positive": int(confidence.nnz),
        "positive_threshold": POSITIVE_THRESHOLD,
        "als": {
            "factors": args.factors,
            "regularisation": args.regularisation,
            "iterations": args.iterations,
            "alpha": args.alpha,
        },
        "knn": {"top_k": args.top_k, "items_with_neighbours": reachable},
        "content": {
            "model": content_mod.MODEL_NAME,
            "dim": int(embeddings.shape[1]),
            **coverage,
        },
        "tmdb_films_cached": len(tmdb),
        "build_seconds": round(time.perf_counter() - wall, 1),
    }
    (ARTIFACT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    total_bytes = sum(p.stat().st_size for p in ARTIFACT_DIR.iterdir() if p.is_file())
    print(f"\nartifacts -> {ARTIFACT_DIR} ({total_bytes / 1e6:.1f} MB)")
    for path in sorted(ARTIFACT_DIR.iterdir()):
        if path.is_file() and path.name != ".gitkeep":
            print(f"  {path.name:<24} {path.stat().st_size / 1e6:>7.2f} MB")
    print(f"\ntotal build time {manifest['build_seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
