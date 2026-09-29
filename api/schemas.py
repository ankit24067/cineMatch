"""Request and response models.

These are the API's contract. The internal `Recommendation` dataclass speaks in
item *indices*; everything here speaks in MovieLens `movie_id`, because an index
is an implementation detail of how the artifacts were built and would break the
moment the catalogue is rebuilt from a different dataset edition.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

MAX_SEEDS = 50
MAX_K = 50


class Movie(BaseModel):
    """A catalogue entry, as returned by search and detail endpoints."""

    movie_id: int
    title: str
    year: int | None = None
    genres: list[str] = Field(default_factory=list)
    poster: str | None = None
    overview: str | None = None


class Explanation(BaseModel):
    """Why this film was surfaced — the actual scoring breakdown.

    `contributions` are fractions of the final score and sum to 1.0 (within
    float tolerance) whenever the score is non-zero. They are captured during
    scoring rather than reconstructed afterwards, so they cannot disagree with
    the ranking.
    """

    contributions: dict[str, float]
    because_of: Movie | None = Field(
        default=None,
        description="The seed film that contributed most, for 'Because you watched X'.",
    )
    seed_similarity: float = 0.0


class Recommendation(BaseModel):
    movie: Movie
    score: float
    rank: int
    explanation: Explanation


class RecommendRequest(BaseModel):
    """Seeds drive the cold-start path; `user_id` drives the warm path.

    At least one must be supplied. When both are, the seeds win and the user id
    is used only to exclude films they have already seen — that is what lets the
    UI say "recommend from these five" for an existing account.
    """

    seed_movie_ids: list[int] = Field(default_factory=list, max_length=MAX_SEEDS)
    user_id: int | None = None
    k: int = Field(default=10, ge=1, le=MAX_K)
    mmr_lambda: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Relevance/diversity tradeoff. 1.0 disables diversification.",
    )
    exclude_seen: bool = True


class RecommendResponse(BaseModel):
    recommendations: list[Recommendation]
    seeds: list[Movie]
    personalised: bool = Field(
        description="False when we had too little signal to claim personalisation.",
    )
    cold_start: bool
    took_ms: float


class SimilarResponse(BaseModel):
    movie: Movie
    similar: list[Movie]
    similarities: list[float]


class SearchResponse(BaseModel):
    """Semantic search results, most similar first.

    `scores` are cosines in the MiniLM content space, so they are comparable
    across queries but are not probabilities. A query no film sits near still
    returns its k nearest neighbours, at correspondingly low scores.
    """

    query: str
    results: list[Movie]
    scores: list[float]
    took_ms: float


class HealthResponse(BaseModel):
    status: str
    n_items: int
    n_users: int
    has_interactions: bool = Field(
        description="False in deployments without the raw ratings; cold start only.",
    )
    artifacts_built_at: str | None = None
    has_semantic_search: bool = Field(
        description="False when the exported encoder is absent; /search returns 503.",
    )
    weights: dict[str, float]
    mmr_lambda: float
