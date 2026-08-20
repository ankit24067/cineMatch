"""Fetch plot overviews and poster paths from TMDB, once, into a local cache.

Why this exists: MovieLens ships no synopses, so without TMDB the content
document for a film is roughly "Goodfellas. released 1990. a crime drama film."
— 65 characters, mostly genre labels. Overviews give the embedding model real
semantic material, which is what makes natural-language search work.

The output is committed to `artifacts/tmdb.json`, so **the deployed service
never calls TMDB and needs no API key**. This script runs on a developer
machine only.

    export TMDB_API_KEY=...        # or put it in .env
    python scripts/fetch_tmdb.py

The run is resumable: results are flushed to disk periodically and already
fetched ids are skipped, so an interrupted run costs nothing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
ARTIFACT_DIR = REPO_ROOT / "artifacts"
CACHE_PATH = ARTIFACT_DIR / "tmdb.json"

API_ROOT = "https://api.themoviedb.org/3/movie"
IMAGE_BASE = "https://image.tmdb.org/t/p/w342"

REQUESTS_PER_SECOND = 20
FLUSH_EVERY = 200


def load_api_key() -> str:
    """Read TMDB_API_KEY from the environment, falling back to a .env file."""
    key = os.environ.get("TMDB_API_KEY", "").strip()
    if key:
        return key

    env_path = REPO_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("TMDB_API_KEY") and "=" in line:
                return line.split("=", 1)[1].strip().strip("\"'")

    raise SystemExit(
        "TMDB_API_KEY is not set.\n"
        "  Get a free key at https://www.themoviedb.org/settings/api\n"
        "  then either:\n"
        "    export TMDB_API_KEY=your_key      (PowerShell: $env:TMDB_API_KEY='your_key')\n"
        "  or write it into a .env file at the repo root (gitignored):\n"
        "    TMDB_API_KEY=your_key"
    )


def load_cache() -> dict[str, dict]:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    return {}


def save_cache(cache: dict[str, dict]) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, separators=(",", ":")), "utf-8")


def fetch_one(session: requests.Session, tmdb_id: int, api_key: str) -> dict | None:
    """Return the fields we keep, or None if TMDB has no record.

    Only four fields are retained. The cache ships in the repository, so there
    is no reason to store the full payload.
    """
    try:
        response = session.get(
            f"{API_ROOT}/{tmdb_id}",
            params={"api_key": api_key, "language": "en-US"},
            timeout=15,
        )
    except requests.RequestException:
        return None

    if response.status_code == 404:
        return {"missing": True}
    if response.status_code == 429:
        time.sleep(float(response.headers.get("Retry-After", 2)))
        return None
    if response.status_code == 401:
        raise SystemExit("TMDB rejected the API key (401). Check TMDB_API_KEY.")
    if not response.ok:
        return None

    payload = response.json()
    poster = payload.get("poster_path")
    return {
        "overview": (payload.get("overview") or "").strip(),
        "poster": f"{IMAGE_BASE}{poster}" if poster else "",
        "tagline": (payload.get("tagline") or "").strip(),
        "runtime": payload.get("runtime") or 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0, help="fetch at most N films (0 = all)")
    parser.add_argument("--refresh", action="store_true", help="ignore the existing cache")
    args = parser.parse_args()

    api_key = load_api_key()

    links = pd.read_csv(RAW_DIR / "links.csv")
    links = links.dropna(subset=["tmdbId"])
    links["tmdbId"] = links["tmdbId"].astype(int)

    cache = {} if args.refresh else load_cache()
    pending = [
        (int(row.movieId), int(row.tmdbId))
        for row in links.itertuples()
        if str(int(row.movieId)) not in cache
    ]
    if args.limit:
        pending = pending[: args.limit]

    print(f"cache holds {len(cache)} films; {len(pending)} to fetch")
    if not pending:
        print("nothing to do")
        return 0

    delay = 1.0 / REQUESTS_PER_SECOND
    failures = 0

    with requests.Session() as session:
        for n, (movie_id, tmdb_id) in enumerate(tqdm(pending, unit="film"), start=1):
            record = fetch_one(session, tmdb_id, api_key)
            if record is None:
                failures += 1
            else:
                cache[str(movie_id)] = record
            if n % FLUSH_EVERY == 0:
                save_cache(cache)
            time.sleep(delay)

    save_cache(cache)

    with_overview = sum(1 for v in cache.values() if v.get("overview"))
    with_poster = sum(1 for v in cache.values() if v.get("poster"))
    total = len(links)
    print(f"\ncached {len(cache)} films -> {CACHE_PATH}")
    print(f"  overviews: {with_overview} ({with_overview / total * 100:.1f}% of catalogue)")
    print(f"  posters:   {with_poster} ({with_poster / total * 100:.1f}% of catalogue)")
    print(f"  size:      {CACHE_PATH.stat().st_size / 1e6:.1f} MB")
    if failures:
        print(f"  {failures} transient failures — re-run to retry them", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
