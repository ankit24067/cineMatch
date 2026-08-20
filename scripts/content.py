"""Content representation: one text document per film, then a dense embedding.

The content side carries three jobs the collaborative models cannot do:

* reach the 20 films with no usable collaborative signal (18 unrated, 2 whose
  single rating centres to zero),
* build a taste profile for a user with no history at all (cold start), and
* supply the similarity term MMR uses to diversify a slate.

Text is assembled from data we actually have — MovieLens ships no plot
overviews, so the document is `title · year · genres · top tags`. Tags exist for
only a minority of films; genres are the floor for everything else.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384

_YEAR = re.compile(r"\((\d{4})\)\s*$")

MAX_TAGS_PER_FILM = 12


def _split_title_year(title: str) -> tuple[str, str]:
    match = _YEAR.search(title.strip())
    if not match:
        return title.strip(), ""
    return title[: match.start()].strip(), match.group(1)


def build_documents(
    movies: pd.DataFrame,
    tags: pd.DataFrame,
    item_ids: np.ndarray,
    tmdb: dict[str, dict] | None = None,
) -> tuple[list[str], dict[str, float]]:
    """One document per item, ordered to match `item_ids`.

    `tmdb` is the optional cache from `scripts/fetch_tmdb.py`, keyed by movieId
    as a string. When present it contributes the plot overview and tagline,
    which roughly quadruples the average document length and is what makes
    natural-language search meaningful.

    Returns the documents and a coverage report for the build log.
    """
    indexed = movies.set_index("movieId")

    tag_counts = (
        tags.assign(tag=tags["tag"].astype(str).str.strip().str.lower())
        .groupby(["movieId", "tag"])
        .size()
        .reset_index(name="n")
        .sort_values(["movieId", "n"], ascending=[True, False])
    )
    tags_by_film: dict[int, list[str]] = (
        tag_counts.groupby("movieId")["tag"].apply(lambda s: list(s)[:MAX_TAGS_PER_FILM]).to_dict()
    )

    tmdb = tmdb or {}
    documents: list[str] = []
    n_tagged = 0
    n_genreless = 0
    n_overview = 0

    for movie_id in item_ids:
        row = indexed.loc[movie_id]
        title, year = _split_title_year(str(row["title"]))

        raw_genres = str(row["genres"])
        if raw_genres == "(no genres listed)":
            genres: list[str] = []
            n_genreless += 1
        else:
            genres = raw_genres.split("|")

        film_tags = tags_by_film.get(int(movie_id), [])
        if film_tags:
            n_tagged += 1

        parts = [title]
        if year:
            parts.append(f"released {year}")
        if genres:
            parts.append(f"a {' '.join(g.lower() for g in genres)} film")

        record = tmdb.get(str(int(movie_id)), {})
        tagline = record.get("tagline", "")
        overview = record.get("overview", "")
        if tagline:
            parts.append(tagline.rstrip("."))
        if overview:
            parts.append(overview[:600])
            n_overview += 1

        if film_tags:
            parts.append(f"themes: {', '.join(film_tags)}")

        documents.append(". ".join(parts) + ".")

    coverage = {
        "n_documents": float(len(documents)),
        "pct_with_tags": round(n_tagged / len(documents) * 100, 1),
        "pct_with_overview": round(n_overview / len(documents) * 100, 1),
        "pct_genreless": round(n_genreless / len(documents) * 100, 1),
        "mean_chars": round(float(np.mean([len(d) for d in documents])), 1),
    }
    return documents, coverage


def encode(documents: list[str], batch_size: int = 64, verbose: bool = True) -> np.ndarray:
    """Encode documents to L2-normalised MiniLM vectors.

    Imported lazily: torch and sentence-transformers are development-only
    dependencies and must never be a requirement of the serving image.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL_NAME, device="cpu")
    vectors = model.encode(
        documents,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=verbose,
    )
    return vectors.astype(np.float32)


def popularity_table(ratings: pd.DataFrame, item_ids: np.ndarray) -> pd.DataFrame:
    """Per-item popularity, a damped quality score, and novelty.

    A raw mean rating puts a single 5.0 above a film with 300 ratings averaging
    4.4, so quality is shrunk toward the global mean in proportion to how little
    evidence a film has (a Bayesian/IMDb-style prior).

    `novelty` is −log₂(p(item)), the self-information of recommending it: high
    for obscure films, low for blockbusters. The evaluation averages it over a
    slate to measure whether we only surface the obvious.
    """
    grouped = ratings.groupby("movieId")["rating"]
    counts = grouped.size()
    means = grouped.mean()

    total = len(ratings)
    global_mean = float(ratings["rating"].mean())
    prior = float(counts.median())

    table = pd.DataFrame(index=pd.Index(item_ids, name="movieId"))
    table["n_ratings"] = counts.reindex(item_ids).fillna(0).astype(int)
    table["mean_rating"] = means.reindex(item_ids).astype(float)

    n = table["n_ratings"].to_numpy(dtype=float)
    m = table["mean_rating"].fillna(global_mean).to_numpy(dtype=float)
    table["damped_rating"] = (n * m + prior * global_mean) / (n + prior)

    share = np.divide(n, total, out=np.zeros_like(n), where=total > 0)
    floor = 1.0 / total
    table["novelty"] = -np.log2(np.maximum(share, floor))

    log_count = np.log1p(n)
    table["popularity"] = log_count / log_count.max() if log_count.max() > 0 else log_count

    return table.reset_index()
