"""Leakage-free temporal splitting.

A random train/test split lets a model observe a user's future while predicting
their past, which inflates every metric. Every number we report comes from the
split below instead: per user, sort by timestamp, train on the earliest 80%,
test on the most recent 20%.

Three-way (train/validation/test) is used for tuning so that hyperparameters are
never chosen on the numbers we later report.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from matrix import POSITIVE_THRESHOLD, IdMaps
from scipy import sparse


@dataclass
class Split:
    """Training interactions plus per-user held-out relevant items."""

    train: sparse.csr_matrix
    # user index -> item indices the user rated >= threshold in the held-out period
    truth: dict[int, np.ndarray]
    name: str

    @property
    def n_evaluable(self) -> int:
        return len(self.truth)


def temporal_split(
    ratings: pd.DataFrame,
    maps: IdMaps,
    test_fraction: float = 0.2,
    validation_fraction: float = 0.0,
    min_interactions: int = 5,
    threshold: float = POSITIVE_THRESHOLD,
) -> tuple[Split, Split | None]:
    """Split per user by time.

    Returns (test_split, validation_split). When `validation_fraction` is 0 the
    second element is None.

    The validation slice is carved from the *end of training*, immediately
    before the test period, so tuning sees the same kind of temporal gap that
    the final evaluation does.

    A user contributes to `truth` only if their held-out period contains at
    least one item rated >= `threshold`; a user whose future contains nothing
    they liked cannot produce a meaningful ranking score, and averaging a
    guaranteed zero over such users would depress every metric uniformly
    without telling us anything about quality.
    """
    user_index, item_index = maps.user_index, maps.item_index

    frame = ratings.copy()
    frame["u"] = frame["userId"].map(user_index)
    frame["i"] = frame["movieId"].map(item_index)
    frame = frame.dropna(subset=["u", "i"])
    frame["u"] = frame["u"].astype(np.int64)
    frame["i"] = frame["i"].astype(np.int64)

    # Stable ordering: timestamp ties are common (bulk-imported sessions), so
    # break them by movieId to keep the split deterministic across runs.
    frame = frame.sort_values(["u", "timestamp", "movieId"], kind="mergesort")

    train_rows: list[np.ndarray] = []
    test_truth: dict[int, np.ndarray] = {}
    validation_rows: list[np.ndarray] = []
    validation_truth: dict[int, np.ndarray] = {}

    for user, group in frame.groupby("u", sort=False):
        n = len(group)
        if n < min_interactions:
            # Too short to hold out from: keep entirely in training.
            train_rows.append(group.to_numpy())
            continue

        n_test = max(1, int(round(n * test_fraction)))
        head, tail = group.iloc[: n - n_test], group.iloc[n - n_test :]

        relevant = tail.loc[tail["rating"] >= threshold, "i"].to_numpy()
        if relevant.size:
            test_truth[int(user)] = np.unique(relevant)

        if validation_fraction > 0 and len(head) >= min_interactions:
            n_val = max(1, int(round(n * validation_fraction)))
            n_val = min(n_val, len(head) - 1)
            inner_head, inner_tail = head.iloc[: len(head) - n_val], head.iloc[len(head) - n_val :]
            val_relevant = inner_tail.loc[inner_tail["rating"] >= threshold, "i"].to_numpy()
            if val_relevant.size:
                validation_truth[int(user)] = np.unique(val_relevant)
            validation_rows.append(inner_head.to_numpy())
        else:
            validation_rows.append(head.to_numpy())

        train_rows.append(head.to_numpy())

    columns = list(frame.columns)

    def to_matrix(rows: list[np.ndarray]) -> sparse.csr_matrix:
        if not rows:
            return sparse.csr_matrix((maps.n_users, maps.n_items), dtype=np.float32)
        stacked = pd.DataFrame(np.vstack(rows), columns=columns)
        matrix = sparse.csr_matrix(
            (
                stacked["rating"].to_numpy(dtype=np.float32),
                (stacked["u"].to_numpy(dtype=np.int64), stacked["i"].to_numpy(dtype=np.int64)),
            ),
            shape=(maps.n_users, maps.n_items),
        )
        matrix.sort_indices()
        return matrix

    test_split = Split(train=to_matrix(train_rows), truth=test_truth, name="test")
    if validation_fraction <= 0:
        return test_split, None

    validation_split = Split(
        train=to_matrix(validation_rows), truth=validation_truth, name="validation"
    )
    return test_split, validation_split


def cohort_of(train: sparse.csr_matrix, user: int) -> str:
    """Bucket a user by training-history length, for per-cohort reporting.

    MovieLens guarantees >= 20 ratings per user, so "cold" here means cold
    *relative to this dataset* — there are no genuinely new users to measure.
    """
    n = int(train.indptr[user + 1] - train.indptr[user])
    if n < 25:
        return "light"
    if n < 100:
        return "medium"
    return "heavy"
