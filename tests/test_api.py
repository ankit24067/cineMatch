"""Endpoint behaviour, including the properties the README claims."""

from __future__ import annotations

import pytest


def test_health_reports_loaded_artifacts(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["n_items"] == 9742
    assert body["n_users"] == 610
    assert set(body["weights"]) == {"als", "item_cf", "content", "popularity"}


def test_search_finds_by_substring_case_insensitively(client):
    body = client.get("/movies", params={"q": "toy story"}).json()
    assert body, "expected at least one Toy Story"
    assert all("toy story" in m["title"].lower() for m in body)


def test_search_ranks_prefix_matches_first(client):
    body = client.get("/movies", params={"q": "star wars", "limit": 5}).json()
    assert body[0]["title"].lower().startswith("star wars")


def test_search_respects_limit(client):
    assert len(client.get("/movies", params={"q": "the", "limit": 3}).json()) == 3


def test_movie_detail_carries_tmdb_metadata(client, seed_ids):
    body = client.get(f"/movies/{seed_ids[0]}").json()
    assert body["movie_id"] == seed_ids[0]
    assert body["genres"], "expected genres"
    assert body["overview"], "expected a TMDB overview on a well-known film"
    assert body["poster"], "expected a TMDB poster on a well-known film"


def test_unknown_movie_is_404(client):
    assert client.get("/movies/99999999").status_code == 404


def test_similar_returns_requested_count(client, seed_ids):
    body = client.get(f"/movies/{seed_ids[0]}/similar", params={"k": 8}).json()
    assert len(body["similar"]) == 8
    assert len(body["similarities"]) == 8
    assert body["similarities"] == sorted(body["similarities"], reverse=True)
    assert seed_ids[0] not in [m["movie_id"] for m in body["similar"]]


class TestRecommend:
    def test_cold_start_returns_k_ranked_items(self, client, seed_ids):
        body = client.post("/recommend", json={"seed_movie_ids": seed_ids, "k": 10}).json()
        assert len(body["recommendations"]) == 10
        assert body["cold_start"] is True
        assert body["personalised"] is True
        assert [r["rank"] for r in body["recommendations"]] == list(range(1, 11))

    def test_seeds_are_never_recommended_back(self, client, seed_ids):
        body = client.post("/recommend", json={"seed_movie_ids": seed_ids, "k": 20}).json()
        returned = {r["movie"]["movie_id"] for r in body["recommendations"]}
        assert returned.isdisjoint(set(seed_ids))

    def test_no_duplicate_recommendations(self, client, seed_ids):
        body = client.post("/recommend", json={"seed_movie_ids": seed_ids, "k": 25}).json()
        ids = [r["movie"]["movie_id"] for r in body["recommendations"]]
        assert len(ids) == len(set(ids))

    def test_scores_are_descending_without_diversification(self, client, seed_ids):
        """With lambda = 1.0 MMR degenerates to plain relevance ordering."""
        body = client.post(
            "/recommend", json={"seed_movie_ids": seed_ids, "k": 10, "mmr_lambda": 1.0}
        ).json()
        scores = [r["score"] for r in body["recommendations"]]
        assert scores == sorted(scores, reverse=True)

    def test_explanation_contributions_sum_to_one(self, client, seed_ids):
        """The README claims the breakdown is the real one, not a reconstruction."""
        body = client.post("/recommend", json={"seed_movie_ids": seed_ids, "k": 10}).json()
        for item in body["recommendations"]:
            total = sum(item["explanation"]["contributions"].values())
            assert total == pytest.approx(1.0, abs=1e-4), item["movie"]["title"]

    def test_cold_start_has_no_als_contribution(self, client, seed_ids):
        """user = -1 has no latent factors, so ALS must contribute nothing."""
        body = client.post("/recommend", json={"seed_movie_ids": seed_ids, "k": 10}).json()
        for item in body["recommendations"]:
            assert item["explanation"]["contributions"]["als"] == 0.0

    def test_diversification_lowers_intra_list_similarity(self, client, seed_ids):
        """MMR should widen the slate relative to pure relevance ranking."""
        payload = {"seed_movie_ids": seed_ids, "k": 10}
        diverse = client.post("/recommend", json={**payload, "mmr_lambda": 0.5}).json()
        greedy = client.post("/recommend", json={**payload, "mmr_lambda": 1.0}).json()

        def genre_count(body):
            return len({g for r in body["recommendations"] for g in r["movie"]["genres"]})

        assert genre_count(diverse) >= genre_count(greedy)

    def test_single_seed_is_not_called_personalised(self, client, seed_ids):
        """Below COLD_START_MINIMUM we stop claiming the result is personalised."""
        body = client.post("/recommend", json={"seed_movie_ids": seed_ids[:1], "k": 5}).json()
        assert body["personalised"] is False

    def test_empty_request_is_rejected(self, client):
        assert client.post("/recommend", json={"k": 10}).status_code == 422

    def test_unknown_seed_is_404(self, client):
        response = client.post("/recommend", json={"seed_movie_ids": [99999999]})
        assert response.status_code == 404

    def test_k_above_limit_is_rejected(self, client, seed_ids):
        response = client.post("/recommend", json={"seed_movie_ids": seed_ids, "k": 500})
        assert response.status_code == 422

    def test_duplicate_seeds_do_not_change_the_slate(self, client, seed_ids):
        once = client.post("/recommend", json={"seed_movie_ids": seed_ids, "k": 10}).json()
        twice = client.post(
            "/recommend", json={"seed_movie_ids": seed_ids + seed_ids, "k": 10}
        ).json()
        assert [r["movie"]["movie_id"] for r in once["recommendations"]] == [
            r["movie"]["movie_id"] for r in twice["recommendations"]
        ]


def test_evaluation_endpoint_exposes_measured_results(client):
    body = client.get("/evaluation").json()
    names = {r["name"] for r in body["results"]}
    assert "Hybrid + MMR (shipped)" in names
    assert "config" in body
