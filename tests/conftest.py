"""Shared fixtures.

The API is exercised against the *real* artifacts rather than fabricated ones.
They are committed to the repository and load in about a second, and a fixture
built from synthetic embeddings would pass while the shipped artifacts were
broken — which is the failure these tests exist to catch.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.artifacts import get_service
from api.main import app


@pytest.fixture(scope="session")
def service():
    return get_service()


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def seed_ids(service) -> list[int]:
    """Five well-known films that are certain to be in MovieLens."""
    wanted = ["Toy Story", "Matrix, The", "Pulp Fiction", "Godfather, The", "Fight Club"]
    frame = service.catalogue.frame
    ids: list[int] = []
    for title in wanted:
        hits = frame[frame["title"].str.contains(title, regex=False)]
        if len(hits):
            ids.append(int(hits.iloc[0]["movie_id"]))
    assert len(ids) >= 3, f"expected the fixture films in the catalogue, found {ids}"
    return ids
