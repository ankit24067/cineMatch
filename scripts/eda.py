"""Dataset statistics for README section 6 and the EDA notebook.

Kept as an importable module rather than living only in the notebook so the
numbers quoted in the documentation are reproducible from the command line:

    python scripts/eda.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"


def load_raw() -> dict[str, pd.DataFrame]:
    """Load the four MovieLens CSVs, with a useful error if they are absent."""
    if not (RAW_DIR / "ratings.csv").exists():
        raise FileNotFoundError(f"{RAW_DIR} is empty. Run `python scripts/download_data.py` first.")
    return {
        "ratings": pd.read_csv(RAW_DIR / "ratings.csv"),
        "movies": pd.read_csv(RAW_DIR / "movies.csv"),
        "tags": pd.read_csv(RAW_DIR / "tags.csv"),
        "links": pd.read_csv(RAW_DIR / "links.csv"),
    }


def summarise(data: dict[str, pd.DataFrame]) -> dict[str, object]:
    """Compute the headline figures the README quotes."""
    ratings, movies, tags = data["ratings"], data["movies"], data["tags"]

    n_users = ratings["userId"].nunique()
    n_rated_movies = ratings["movieId"].nunique()
    n_catalogue = len(movies)
    n_ratings = len(ratings)

    density = n_ratings / (n_users * n_catalogue) * 100

    timestamps = pd.to_datetime(ratings["timestamp"], unit="s")
    per_user = ratings.groupby("userId").size()
    per_item = ratings.groupby("movieId").size()

    ordered = per_item.sort_values(ascending=False).to_numpy()
    cumulative = np.cumsum(ordered) / ordered.sum()
    top_1pct = ordered[: max(1, len(ordered) // 100)].sum() / ordered.sum() * 100
    items_for_half = int(np.searchsorted(cumulative, 0.5) + 1)

    return {
        "n_ratings": n_ratings,
        "n_users": n_users,
        "n_catalogue": n_catalogue,
        "n_rated_movies": n_rated_movies,
        "n_unrated_movies": n_catalogue - n_rated_movies,
        "n_tags": len(tags),
        "density_pct": round(density, 4),
        "rating_min": ratings["rating"].min(),
        "rating_max": ratings["rating"].max(),
        "rating_mean": round(ratings["rating"].mean(), 3),
        "pct_positive": round((ratings["rating"] >= 4.0).mean() * 100, 1),
        "first_rating": str(timestamps.min().date()),
        "last_rating": str(timestamps.max().date()),
        "ratings_per_user_median": int(per_user.median()),
        "ratings_per_user_min": int(per_user.min()),
        "ratings_per_item_median": int(per_item.median()),
        "pct_items_under_5_ratings": round((per_item < 5).mean() * 100, 1),
        "top_1pct_share": round(top_1pct, 1),
        "items_for_half_of_ratings": items_for_half,
    }


def main() -> int:
    stats = summarise(load_raw())
    width = max(len(k) for k in stats)
    for key, value in stats.items():
        print(f"{key:<{width}}  {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
