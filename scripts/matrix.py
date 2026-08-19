"""Shared data loading and index mapping for the offline pipeline.

The recommender works in *dense contiguous index space* (0..n_users-1,
0..n_items-1) while MovieLens uses sparse, non-contiguous ids. Every artifact is
written in index space, and `IdMaps` is the single source of truth for the
translation. It is saved alongside the artifacts so the serving layer maps ids
identically without re-reading the raw CSVs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
ARTIFACT_DIR = REPO_ROOT / "artifacts"

POSITIVE_THRESHOLD = 4.0


@dataclass
class IdMaps:
    """Bidirectional MovieLens id <-> contiguous index mappings."""

    user_ids: np.ndarray  # index -> MovieLens userId
    item_ids: np.ndarray  # index -> MovieLens movieId

    @property
    def n_users(self) -> int:
        return len(self.user_ids)

    @property
    def n_items(self) -> int:
        return len(self.item_ids)

    @property
    def user_index(self) -> dict[int, int]:
        return {int(u): i for i, u in enumerate(self.user_ids)}

    @property
    def item_index(self) -> dict[int, int]:
        return {int(m): i for i, m in enumerate(self.item_ids)}

    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps(
                {
                    "user_ids": [int(u) for u in self.user_ids],
                    "item_ids": [int(m) for m in self.item_ids],
                }
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> IdMaps:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            user_ids=np.asarray(payload["user_ids"], dtype=np.int64),
            item_ids=np.asarray(payload["item_ids"], dtype=np.int64),
        )


def load_frames() -> dict[str, pd.DataFrame]:
    if not (RAW_DIR / "ratings.csv").exists():
        raise FileNotFoundError(f"{RAW_DIR} is empty. Run `python scripts/download_data.py` first.")
    return {
        "ratings": pd.read_csv(RAW_DIR / "ratings.csv"),
        "movies": pd.read_csv(RAW_DIR / "movies.csv"),
        "tags": pd.read_csv(RAW_DIR / "tags.csv"),
        "links": pd.read_csv(RAW_DIR / "links.csv"),
    }


def build_id_maps(ratings: pd.DataFrame, movies: pd.DataFrame) -> IdMaps:
    """Index every user, and every film in the *catalogue*.

    Items deliberately span all of movies.csv rather than only rated films: 18
    films carry no ratings at all, and the content model must still be able to
    reach them (README section 15, item cold start).
    """
    return IdMaps(
        user_ids=np.sort(ratings["userId"].unique()),
        item_ids=np.sort(movies["movieId"].unique()),
    )


def build_rating_matrix(ratings: pd.DataFrame, maps: IdMaps) -> sparse.csr_matrix:
    """Raw ratings as a sparse user x item matrix, values on the 0.5-5.0 scale."""
    user_index, item_index = maps.user_index, maps.item_index
    rows = ratings["userId"].map(user_index).to_numpy()
    cols = ratings["movieId"].map(item_index).to_numpy()

    # A rating can reference a movieId absent from movies.csv in some MovieLens
    # editions; drop those rather than silently mis-indexing.
    valid = ~pd.isna(cols)
    if not valid.all():
        print(f"  ! dropping {(~valid).sum()} ratings with unknown movieId")
        rows, cols = rows[valid], cols[valid]
        values = ratings["rating"].to_numpy()[valid]
    else:
        values = ratings["rating"].to_numpy()

    matrix = sparse.csr_matrix(
        (values.astype(np.float32), (rows.astype(np.int64), cols.astype(np.int64))),
        shape=(maps.n_users, maps.n_items),
    )
    matrix.sort_indices()
    return matrix


def build_confidence_matrix(
    matrix: sparse.csr_matrix, alpha: float = 40.0, threshold: float = POSITIVE_THRESHOLD
) -> sparse.csr_matrix:
    """Implicit-feedback formulation for ALS.

    Ratings >= `threshold` become positive interactions with confidence
    `c = 1 + alpha * r`, where r is the rating's margin above the threshold
    normalised to [0, 1]. Sub-threshold ratings are dropped rather than treated
    as negatives: absence means *unseen*, not *disliked* (README assumption 2).
    """
    kept = matrix.copy()
    kept.data = np.where(kept.data >= threshold, kept.data, 0.0)
    kept.eliminate_zeros()

    margin = (kept.data - threshold) / (5.0 - threshold)  # 4.0 -> 0.0, 5.0 -> 1.0
    kept.data = (1.0 + alpha * margin).astype(np.float32)
    return kept


def mean_center_rows(matrix: sparse.csr_matrix) -> sparse.csr_matrix:
    """Subtract each user's mean rating from their observed entries.

    Removes per-user rating bias (some users never rate below 3) so that item
    similarity reflects agreement in *relative* preference, not absolute
    generosity. Only observed cells are touched; zeros stay structural.
    """
    centred = matrix.copy().astype(np.float32)
    counts = np.diff(centred.indptr)
    sums = np.add.reduceat(centred.data, centred.indptr[:-1]) if centred.nnz else np.zeros(0)
    # reduceat misbehaves on empty rows; guard by zeroing their contribution.
    sums = np.where(counts > 0, sums, 0.0)
    means = np.divide(sums, counts, out=np.zeros_like(sums, dtype=np.float64), where=counts > 0)
    centred.data = centred.data - np.repeat(means, counts).astype(np.float32)
    return centred


def l2_normalise_columns(matrix: sparse.csr_matrix) -> sparse.csc_matrix:
    """Scale each item column to unit norm so a dot product is a cosine."""
    csc = matrix.tocsc().astype(np.float32)
    norms = np.sqrt(csc.multiply(csc).sum(axis=0)).A1
    norms[norms == 0] = 1.0
    scale = sparse.diags(1.0 / norms, dtype=np.float32)
    return (csc @ scale).tocsc()
