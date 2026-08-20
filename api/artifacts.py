"""Artifact loading and id translation for the serving layer.

Loaded exactly once, at process start, and then treated as immutable. Nothing
here trains and nothing re-reads the raw dataset on the request path.

Two deployment modes, distinguished by whether `data/raw/` exists:

    with raw data     (local dev)  full interaction matrix; warm users work
    without raw data  (deployed)   empty matrix; cold start only

The MovieLens licence forbids redistributing the ratings, so `data/raw/` is
gitignored and the deployed container genuinely does not have it (README §6).
The derived artifacts *are* shipped, which is why every expensive quantity was
precomputed offline. In the deployed mode the API serves the cold-start path —
the caller supplies seed films and we pass `user = -1` — which is the flow the
demo UI uses anyway.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from api.recommender import DEFAULT_LAMBDA, Components, Recommender

REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = REPO_ROOT / "artifacts"
RAW_DIR = REPO_ROOT / "data" / "raw"

# A film that appears in no user's history has no collaborative signal; the
# recommender falls back to content for these (see Recommender.similar).
COLD_ITEM_NEIGHBOURS = 0


@dataclass(frozen=True)
class Catalogue:
    """Catalogue rows plus the id <-> index translation.

    `Recommender` works entirely in contiguous index space. Every id crossing
    the HTTP boundary is a MovieLens `movie_id`, and this is the only place the
    two are allowed to meet.
    """

    frame: pd.DataFrame  # indexed by item index, ordered 0..n_items-1
    item_ids: np.ndarray  # index -> movie_id
    user_ids: np.ndarray  # index -> user_id
    _item_index: dict[int, int]
    _user_index: dict[int, int]

    @property
    def n_items(self) -> int:
        return len(self.item_ids)

    @property
    def n_users(self) -> int:
        return len(self.user_ids)

    def index_of_movie(self, movie_id: int) -> int | None:
        return self._item_index.get(int(movie_id))

    def index_of_user(self, user_id: int) -> int | None:
        return self._user_index.get(int(user_id))

    def movie_id_of(self, index: int) -> int:
        return int(self.item_ids[index])

    def record(self, index: int) -> dict:
        """One catalogue row as a plain dict, with empty strings normalised to None."""
        row = self.frame.iloc[int(index)]
        poster = str(row["poster"]) if row["poster"] else ""
        overview = str(row["overview"]) if row["overview"] else ""
        genres = str(row["genres"])
        return {
            "movie_id": int(row["movie_id"]),
            "title": str(row["title"]),
            "year": int(row["year"]) if row["year"] else None,
            "genres": [g for g in genres.split("|") if g and g != "(no genres listed)"],
            "poster": poster or None,
            "overview": overview or None,
        }


@dataclass(frozen=True)
class Service:
    """Everything a request handler needs."""

    recommender: Recommender
    catalogue: Catalogue
    manifest: dict
    has_interactions: bool

    def titles_matching(self, query: str, limit: int) -> list[int]:
        """Case-insensitive substring match over titles, most popular first.

        This backs the seed picker in the UI, so ranking by popularity matters:
        typing "star" should surface Star Wars before an obscure short film.
        Semantic free-text search is a separate concern and needs the ONNX text
        encoder, which this phase does not ship.
        """
        needle = query.strip().lower()
        if not needle:
            return []
        frame = self.catalogue.frame
        hits = frame.index[frame["title"].str.lower().str.contains(needle, regex=False)]
        indices = np.asarray(hits, dtype=np.int64)
        if indices.size == 0:
            return []
        popularity = self.recommender.c.popularity[indices]
        # Exact prefix matches outrank mere substring hits at equal popularity.
        titles = frame["title"].str.lower().to_numpy()[indices]
        prefix = np.array([t.startswith(needle) for t in titles], dtype=np.float32)
        order = np.argsort(-(prefix + popularity))
        return [int(i) for i in indices[order][:limit]]


def _load_interactions(n_users: int, n_items: int) -> tuple[sparse.csr_matrix, bool]:
    """The observed rating matrix, or an empty one when raw data is absent.

    An empty matrix is a correct degradation rather than a stub: `history` and
    `liked` return nothing, `exclude_seen` excludes nothing, and the ALS term is
    zero for `user = -1`. The cold-start path is unaffected because it takes its
    seeds from the caller.
    """
    if not (RAW_DIR / "ratings.csv").exists():
        return sparse.csr_matrix((n_users, n_items), dtype=np.float32), False

    # Imported lazily: scripts/ is not installed in the serving image.
    from scripts.matrix import build_id_maps, build_rating_matrix, load_frames

    frames = load_frames()
    maps = build_id_maps(frames["ratings"], frames["movies"])
    return build_rating_matrix(frames["ratings"], maps), True


def _load_catalogue(artifact_dir: Path) -> Catalogue:
    frame = pd.read_parquet(artifact_dir / "catalogue.parquet")
    # The recommender addresses items positionally, so the frame must be in
    # index order with no gaps before we ever use .iloc on it.
    frame = frame.sort_values("index").reset_index(drop=True)
    expected = np.arange(len(frame), dtype=np.int64)
    if not np.array_equal(frame["index"].to_numpy(dtype=np.int64), expected):
        raise ValueError("catalogue.parquet index column is not contiguous 0..n-1")

    maps = json.loads((artifact_dir / "id_maps.json").read_text(encoding="utf-8"))
    item_ids = np.asarray(maps["item_ids"], dtype=np.int64)
    user_ids = np.asarray(maps["user_ids"], dtype=np.int64)
    if not np.array_equal(item_ids, frame["movie_id"].to_numpy(dtype=np.int64)):
        raise ValueError("id_maps.json and catalogue.parquet disagree on item order")

    return Catalogue(
        frame=frame,
        item_ids=item_ids,
        user_ids=user_ids,
        _item_index={int(m): i for i, m in enumerate(item_ids)},
        _user_index={int(u): i for i, u in enumerate(user_ids)},
    )


def _load_weights(artifact_dir: Path) -> tuple[dict[str, float] | None, float | None]:
    """Prefer tuned weights when a tuning pass has produced them."""
    path = artifact_dir / "tuned_weights.json"
    if not path.exists():
        return None, None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["weights"], payload["lambda"]


@lru_cache(maxsize=1)
def get_service(artifact_dir: Path = ARTIFACT_DIR) -> Service:
    """Load every artifact. Cached — the first call pays the cost, once."""
    catalogue = _load_catalogue(artifact_dir)
    interactions, has_interactions = _load_interactions(catalogue.n_users, catalogue.n_items)
    components = Components.load(artifact_dir, interactions)
    weights, lambda_ = _load_weights(artifact_dir)

    recommender = Recommender(
        components,
        weights=weights,
        mmr_lambda=DEFAULT_LAMBDA if lambda_ is None else lambda_,
    )
    manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))

    return Service(
        recommender=recommender,
        catalogue=catalogue,
        manifest=manifest,
        has_interactions=has_interactions,
    )
