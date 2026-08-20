"""FastAPI application — the serving layer.

Thin by design. Every ranking decision lives in `api/recommender.py`, which the
offline evaluation imports too; this module maps ids, enriches results with
catalogue metadata, and gets out of the way. If you find scoring logic here,
it is in the wrong file.

Artifacts load once during lifespan startup so the first real request does not
pay a ~1s parquet and numpy load.
"""

from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from api.artifacts import ARTIFACT_DIR, Service, get_service
from api.recommender import COLD_START_MINIMUM
from api.schemas import (
    Explanation,
    HealthResponse,
    Movie,
    Recommendation,
    RecommendRequest,
    RecommendResponse,
    SimilarResponse,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_service()
    yield


app = FastAPI(
    title="CineMatch API",
    description="A hybrid, explainable movie recommendation system.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _movie(service: Service, index: int) -> Movie:
    return Movie(**service.catalogue.record(index))


def _resolve_seeds(service: Service, movie_ids: list[int]) -> tuple[np.ndarray, list[int]]:
    """Translate movie ids to indices, reporting the ones we do not carry."""
    indices: list[int] = []
    unknown: list[int] = []
    for movie_id in movie_ids:
        index = service.catalogue.index_of_movie(movie_id)
        if index is None:
            unknown.append(movie_id)
        elif index not in indices:
            indices.append(index)
    return np.asarray(indices, dtype=np.int64), unknown


@app.get("/", include_in_schema=False)
def index() -> dict:
    """Service index.

    Without this, opening the deployed backend URL returns a bare 404, which
    reads as a broken deployment rather than an API with no page at its root.
    """
    return {
        "service": "CineMatch API",
        "version": app.version,
        "docs": "/docs",
        "endpoints": [
            "GET /health",
            "GET /movies?q=&limit=",
            "GET /movies/{movie_id}",
            "GET /movies/{movie_id}/similar?k=",
            "POST /recommend",
            "GET /evaluation",
        ],
    }


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    service = get_service()
    return HealthResponse(
        status="ok",
        n_items=service.catalogue.n_items,
        n_users=service.catalogue.n_users,
        has_interactions=service.has_interactions,
        artifacts_built_at=service.manifest.get("built_at"),
        weights=service.recommender.weights,
        mmr_lambda=service.recommender.mmr_lambda,
    )


@app.get("/movies", response_model=list[Movie])
def search_movies(
    q: str = Query(min_length=1, max_length=100, description="Title substring."),
    limit: int = Query(default=20, ge=1, le=100),
) -> list[Movie]:
    """Title search backing the seed picker. Popularity-ranked, prefix-first."""
    service = get_service()
    return [_movie(service, i) for i in service.titles_matching(q, limit)]


@app.get("/movies/{movie_id}", response_model=Movie)
def get_movie(movie_id: int) -> Movie:
    service = get_service()
    index = service.catalogue.index_of_movie(movie_id)
    if index is None:
        raise HTTPException(status_code=404, detail=f"unknown movie_id {movie_id}")
    return _movie(service, index)


@app.get("/movies/{movie_id}/similar", response_model=SimilarResponse)
def similar_movies(movie_id: int, k: int = Query(default=10, ge=1, le=50)) -> SimilarResponse:
    """Item-to-item neighbours, with a content fallback for cold films."""
    service = get_service()
    index = service.catalogue.index_of_movie(movie_id)
    if index is None:
        raise HTTPException(status_code=404, detail=f"unknown movie_id {movie_id}")

    pairs = service.recommender.similar(index, k)
    return SimilarResponse(
        movie=_movie(service, index),
        similar=[_movie(service, i) for i, _ in pairs],
        similarities=[round(s, 6) for _, s in pairs],
    )


@app.post("/recommend", response_model=RecommendResponse)
def recommend(request: RecommendRequest) -> RecommendResponse:
    """Ranked, explained recommendations.

    Cold start is the default path: pass `seed_movie_ids` and no `user_id`. The
    recommender is called with `user = -1`, which zeroes the ALS term and leaves
    the item-CF, content, and popularity signals to carry the slate.
    """
    service = get_service()
    started = time.perf_counter()

    seeds, unknown = _resolve_seeds(service, request.seed_movie_ids)
    if unknown:
        raise HTTPException(status_code=404, detail=f"unknown movie_id(s): {unknown}")

    user = -1
    if request.user_id is not None:
        resolved = service.catalogue.index_of_user(request.user_id)
        if resolved is None:
            raise HTTPException(status_code=404, detail=f"unknown user_id {request.user_id}")
        if not service.has_interactions:
            raise HTTPException(
                status_code=409,
                detail=(
                    "this deployment has no interaction data; "
                    "supply seed_movie_ids for cold-start recommendations"
                ),
            )
        user = resolved

    if seeds.size == 0 and user < 0:
        raise HTTPException(
            status_code=422, detail="supply seed_movie_ids, user_id, or both"
        )

    seed_override = seeds if seeds.size else None

    results = service.recommender.recommend(
        user=user,
        k=request.k,
        seeds=seed_override,
        exclude_seen=request.exclude_seen,
        mmr_lambda=request.mmr_lambda,
    )

    effective_seeds = seeds if seeds.size else service.recommender.liked(user)
    recommendations = [
        Recommendation(
            movie=_movie(service, result.item),
            score=round(result.score, 6),
            rank=rank,
            explanation=Explanation(
                contributions={k: round(v, 6) for k, v in result.contributions.items()},
                because_of=(
                    _movie(service, result.seed_item) if result.seed_item is not None else None
                ),
                seed_similarity=round(result.seed_similarity, 6),
            ),
        )
        for rank, result in enumerate(results, start=1)
    ]

    return RecommendResponse(
        recommendations=recommendations,
        seeds=[_movie(service, int(i)) for i in effective_seeds],
        personalised=len(effective_seeds) >= COLD_START_MINIMUM,
        cold_start=user < 0,
        took_ms=round((time.perf_counter() - started) * 1000, 3),
    )


@app.get("/evaluation")
def evaluation() -> dict:
    """The measured offline results, so the UI can show them without a rebuild."""
    path = ARTIFACT_DIR / "evaluation.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="evaluation.json not built")
    return json.loads(path.read_text(encoding="utf-8"))
