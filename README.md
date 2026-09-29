# CineMatch — A Hybrid, Explainable Movie Recommendation System

> **Live demo:** ⟨TODO: deployment URL⟩
> **API docs:** ⟨TODO: backend URL⟩/docs
> **Repository:** https://github.com/ankit24067/cineMatch

**New here? Start at the demo link, pick 5 films you like on the landing screen, and hit *Get Recommendations*.** Every recommendation card has a **"Why this?"** button that opens the actual scoring breakdown behind it.

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Use Case & Motivation](#2-use-case--motivation)
3. [Approach](#3-approach)
4. [System Architecture](#4-system-architecture)
5. [Recommendation Methodology](#5-recommendation-methodology)
6. [Dataset](#6-dataset)
7. [Technologies Used](#7-technologies-used)
8. [Getting Started](#8-getting-started)
9. [API Reference](#9-api-reference)
10. [Evaluation Methodology](#10-evaluation-methodology)
11. [Results](#11-results)
12. [Test Cases](#12-test-cases)
13. [Assumptions](#13-assumptions)
14. [Key Design Decisions](#14-key-design-decisions)
15. [Known Limitations](#15-known-limitations)
16. [Comparison with Netflix](#16-comparison-with-netflix)
17. [Future Improvements](#17-future-improvements)
18. [Licensing & Attribution](#18-licensing--attribution)

---

## 1. Problem Statement

Given a user's sparse and implicit viewing history, produce a ranked list of films that is:

- **Relevant** — the user would plausibly watch and enjoy them,
- **Diverse** — not ten variations of the same film,
- **Explainable** — the user can see *why* each item was surfaced,
- **Robust to cold start** — useful output for a user we have never seen before.

The core difficulty is sparsity. In our dataset the user–item matrix is **1.70% dense**, meaning the average user has rated a vanishingly small fraction of the catalog. A system that only memorizes co-occurrence patterns degenerates into recommending blockbusters to everyone.

---

## 2. Use Case & Motivation

Film discovery is a canonical recommendation problem with a real cost of failure: a bad carousel is a user who closes the app. We chose it because it lets us evaluate honestly — the dataset carries explicit ratings *and* timestamps, so we can construct a leakage-free temporal split and report standard ranking metrics rather than hand-waving.

We were also interested in a second question that production systems care about more than benchmarks do: **can a recommender explain itself without a meaningful accuracy penalty?** Section [11](#11-results) reports what that tradeoff actually cost us.

---

## 3. Approach

We built a **four-stage hybrid pipeline** rather than a single model, because no single technique covers the whole input space:

| Stage | What it does | Why |
|---|---|---|
| **1. Candidate generation** | Retrieve ~300 plausible items via item-kNN neighbours, ALS latent similarity, and content embeddings | Cheap recall-oriented pass; scoring 9k items per request is wasteful |
| **2. Scoring** | Weighted blend of collaborative, content, and popularity signals | Collaborative filtering is strong for warm users; content carries cold users |
| **3. Reranking** | Maximal Marginal Relevance (MMR) over content embeddings | Raw relevance ranking produces monocultures |
| **4. Explanation** | Retain per-component contributions and the top contributing seed item | Required for the "Why this?" panel |

All heavy computation is **precomputed offline** into artifacts. The serving layer performs matrix slices and lookups only — it never trains.

---

## 4. System Architecture

```
┌──────────────────────────────────────────────────────────┐
│  OFFLINE  (scripts/, run once)                           │
│                                                          │
│  download_data.py ──▶ data/raw/  (gitignored)            │
│         │                                                │
│         ▼                                                │
│  build_artifacts.py                                      │
│    ├─ item-item kNN  → neighbors_top50.npz               │
│    ├─ ALS training   → user_factors.npy, item_factors.npy│
│    ├─ MiniLM encode  → content_emb.npy                   │
│    ├─ popularity      → popularity.parquet               │
│    └─ TMDB fetch      → posters.json                     │
└──────────────────────────────────────────────────────────┘
                              │  artifacts/
                              ▼
┌──────────────────────────────────────────────────────────┐
│  SERVING  —  FastAPI in Docker (Render, free tier)       │
│                                                          │
│   POST /recommend         (seeds and/or user_id)         │
│   GET  /movies?q=         (title search)                 │
│   GET  /movies/{id}                                      │
│   GET  /movies/{id}/similar                              │
│   GET  /evaluation · GET /health                         │
│                                                          │
│   Explanations are inline on /recommend, not a call.     │
└──────────────────────────────────────────────────────────┘
                              │  JSON
                              ▼
┌──────────────────────────────────────────────────────────┐
│  FRONTEND  —  React + Vite, static build                 │
│   Hero banner · "Because you watched X" rails            │
│   · "Why this?" explanation panel · cold-start picker    │
└──────────────────────────────────────────────────────────┘
```

**Repository layout**

```
.
├── data/raw/              # gitignored — see download_data.py
├── artifacts/             # precomputed model outputs
├── notebooks/
│   └── 01_eda.ipynb
├── scripts/
│   ├── download_data.py
│   ├── build_artifacts.py
│   ├── export_encoder.py  # MiniLM weights -> artifacts/encoder/
│   └── evaluate.py
├── api/
│   ├── main.py            # FastAPI app
│   ├── recommender.py     # scoring + rerank + explain
│   ├── encoder.py         # MiniLM forward pass in numpy, for /search
│   └── schemas.py
├── frontend/              # React + Vite + Tailwind
├── docs/
│   ├── EVALUATION.md
│   └── TEST_CASES.md
└── README.md
```

---

## 5. Recommendation Methodology

### 5.1 Item–item collaborative filtering
Cosine similarity over the L2-normalised, mean-centred user–item matrix. We keep the **top 50 neighbours per item** and discard the rest, which cuts the similarity matrix from **380 MB** (9,742² dense float32) to **1.5 MB** on disk — a 247× reduction. The full-matrix variant was never evaluated end to end, so the accuracy cost of the truncation is asserted, not measured; 50 is a conventional cutoff, and 9,722 of 9,742 items have a full neighbour list at that depth.

### 5.2 Matrix factorisation (ALS)
Alternating Least Squares on the implicit-feedback formulation, with ratings ≥ 4.0 treated as positive signal and confidence weighting `c = 1 + α·r`.

- Factors: **64** · Regularisation: **0.05** · Iterations: **15** · α: **40.0**
- Selected by grid search on the validation split (see `scripts/evaluate.py`).

### 5.3 Content embeddings
Each film is represented as a single text document — `title · genres · top user tags · overview` — encoded with `all-MiniLM-L6-v2` into a 384-dimensional vector. These power content similarity, cold-start profiles, natural-language search, and the MMR diversity term.

### 5.4 Hybrid scoring

```
score(u, i) = w₁·ALS(u,i) + w₂·itemCF(u,i) + w₃·content(u,i) + w₄·popularity(i)
```

Each component is min–max normalised within the candidate set before blending, so no single signal dominates by virtue of its scale. Weights: `w₁=0.40, w₂=0.20, w₃=0.30, w₄=0.10`, selected by grid search over 22 combinations on the validation split against NDCG@10 (`python scripts/evaluate.py --all --tune`, best validation NDCG@10 = 0.1060, 560 users). The tuned weights land in `artifacts/tuned_weights.json`, which the serving layer prefers over the hand-set defaults when present.

### 5.5 Diversity reranking (MMR)

```
MMR(i) = λ · score(u,i) − (1−λ) · max_{j ∈ selected} cos(emb_i, emb_j)
```

with **λ = 0.6**, chosen by sweeping λ on the validation split and taking the most diverse setting that stayed within 5% of the best NDCG:

| λ | 1.0 | 0.9 | 0.8 | 0.7 | **0.6** | 0.5 |
|---|---|---|---|---|---|---|
| NDCG@10 | 0.1060 | 0.1058 | 0.1064 | 0.1052 | **0.1037** | 0.0968 |
| ILD | 0.713 | 0.720 | 0.727 | 0.737 | **0.750** | 0.769 |

λ = 0.5 is where the tradeoff turns bad: diversity gains 0.019 while NDCG drops 6.7%. At 0.6 the accuracy cost is 2.2% off the best for a 0.037 ILD gain — quantified on the test split in [Results](#11-results).

### 5.6 Cold start
A new user selects ≥ 3 films. We average their content embeddings into a pseudo-profile, retrieve content-nearest items, and fold the profile into ALS latent space to blend in collaborative signal. Below 3 selections the system falls back to a popularity ranking, and **the UI says so explicitly** rather than pretending the result is personalised.

### 5.7 Explanations
For each recommendation we return the normalised contribution of each scoring component plus the strongest contributing seed item. The UI renders this as *"Because you watched **Heat** — 68% collaborative, 24% content, 8% popularity."*

---

## 6. Dataset

**MovieLens `ml-latest-small`** (GroupLens Research)

| Property | Value |
|---|---|
| Ratings | 100,836 |
| Users | 610 |
| Movies | 9,742 |
| Tags | 3,683 |
| Rating scale | 0.5 – 5.0 |
| Time span | 1996 – 2018 |
| Density | 1.70% |

**Why this dataset:** it is small enough to deploy on free-tier infrastructure, carries timestamps (enabling a leakage-free temporal split), and includes user-generated tags that give the content model more signal than genres alone.

**Poster imagery** is joined via `links.csv` → TMDB IDs, fetched once offline and cached in `artifacts/posters.json`, so the running service needs no TMDB API key.

> ⚠️ The MovieLens licence does not permit redistribution. Raw data is **not committed**. Run `python scripts/download_data.py` to fetch it.

---

## 7. Technologies Used

| Area | Stack |
|---|---|
| Language | Python 3.11, TypeScript |
| Data | pandas, NumPy, SciPy (sparse) |
| Models | `implicit` (ALS), scikit-learn, sentence-transformers |
| Runtime inference | MiniLM reimplemented in numpy (`api/encoder.py`) for query encoding only; every other path is matrix slices and lookups. No ML runtime is installed — no torch, no ONNX Runtime |
| API | FastAPI, Uvicorn, Pydantic v2 |
| Frontend | React 18, Vite, TypeScript, Tailwind CSS |
| Deployment | Render (Docker) + Render static site or Vercel. Blueprint in [`render.yaml`](render.yaml); guide in [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md). **Config committed and verified, not yet deployed** |
| Testing | pytest, ruff |

---

## 8. Getting Started

### Prerequisites
Python 3.11+, Node 18+, ~2 GB free disk.

### Backend

```bash
git clone https://github.com/ankit24067/cineMatch.git
cd cineMatch

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Serving only (what the deployed container installs):
pip install -r requirements.txt

# Rebuilding artifacts additionally needs the offline pipeline.
# The CPU-only torch build is not on PyPI, so requirements-dev.txt carries an
# --extra-index-url line pointing at PyTorch's index. One command, no prelude:
pip install -r requirements-dev.txt

python scripts/download_data.py     # fetches MovieLens into data/raw/
python scripts/build_artifacts.py   # ~3.5 min — trains and writes artifacts/
uvicorn api.main:app --reload       # http://localhost:8000/docs
```

The artifacts are committed, so **you can skip straight to `uvicorn`** if you only
want to run the API. `download_data.py` and `build_artifacts.py` are needed only
to rebuild them or to enable warm-user requests (see [§9](#9-api-reference)).

### Frontend

```bash
cd frontend
npm install
cp .env.example .env                # set VITE_API_URL=http://localhost:8000
npm run dev                         # http://localhost:5173
```

### Reproducing the evaluation

```bash
python scripts/evaluate.py --all    # writes docs/EVALUATION.md results table
python scripts/evaluate.py --tune   # grid-searches weights on the validation split
```

Without `--all` the script prints the resolved configuration and exits.

### Rebuilding the query encoder

```bash
python -m scripts.export_encoder   # writes artifacts/encoder/ (41 MB)
```

Committed, so this is only needed if the embedding model changes — and if it
does, rebuild `content_emb.npy` in the same pass. A query encoder and a document
matrix from two different models still produce cosines, they are just
meaningless. `tests/test_encoder.py` fails loudly if the two drift apart.

### Running the tests

```bash
pytest                              # exercises the API against the real artifacts
```

### Deploying

```bash
docker build -t cinematch .
docker run -p 8000:8000 cinematch   # http://localhost:8000/health
```

Both services are configured for Render's free tier via the blueprint in
[`render.yaml`](render.yaml). Full walkthrough, including the free-tier cold
start and why `VITE_API_URL` needs a *rebuild* rather than a restart, in
[`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md).

Note that a deployed instance has **no interaction data** — the MovieLens
licence forbids redistributing the ratings — so it serves the cold-start path
only and `/recommend` with a `user_id` returns 409. The frontend never uses
that path, so nothing in the UI is degraded.

---

## 9. API Reference

Interactive docs at `/docs`. Summary:

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/recommend` | Recommendations with explanations — cold start *and* warm users |
| `GET` | `/movies?q={query}&limit=20` | Title search, backing the seed picker |
| `GET` | `/movies/{movie_id}` | Catalogue detail (title, year, genres, poster, overview) |
| `GET` | `/movies/{movie_id}/similar?k=10` | Item-to-item "More like this" |
| `GET` | `/search?q={query}&k=10` | Natural-language semantic search over the content embeddings |
| `GET` | `/evaluation` | The measured offline results from `artifacts/evaluation.json` |
| `GET` | `/health` | Liveness, artifact build time, active weights, and encoder availability |

**One recommend endpoint, two paths.** Passing `seed_movie_ids` is the cold-start
path: the recommender is called with `user = -1`, which zeroes the ALS term and
lets item-CF, content, and popularity carry the slate. Passing `user_id` uses
that user's stored history instead. Supplying both recommends *from the seeds*
while still excluding what the user has already seen. Splitting these into
separate endpoints would have duplicated the same pipeline behind two URLs.

**Explanations are inline, not a separate call.** Every recommendation carries
its own score decomposition, so the "Why this?" panel needs no second request
and cannot drift from the ranking it describes.

<details>
<summary>Example — <code>POST /recommend</code> with <code>{"seed_movie_ids": [1, 296, 2571, 318], "k": 2}</code></summary>

```json
{
  "recommendations": [
    {
      "movie": {
        "movie_id": 2959,
        "title": "Fight Club",
        "year": 1999,
        "genres": ["Action", "Crime", "Drama", "Thriller"],
        "poster": "https://image.tmdb.org/t/p/w342/...",
        "overview": "A ticking-time-bomb insomniac..."
      },
      "score": 0.554,
      "rank": 1,
      "explanation": {
        "contributions": { "als": 0.0, "item_cf": 0.541981, "content": 0.290132, "popularity": 0.167887 },
        "because_of": { "movie_id": 296, "title": "Pulp Fiction", "year": 1994, "genres": ["Comedy", "Crime", "Drama", "Thriller"] },
        "seed_similarity": 0.412
      }
    }
  ],
  "seeds": [ "…the four resolved seed films…" ],
  "personalised": true,
  "cold_start": true,
  "took_ms": 6.03
}
```

`contributions` are fractions of the final score and sum to 1.0. `als` is exactly
zero here because a cold-start caller has no latent factors — the explanation
reports that honestly rather than hiding the missing signal.
</details>

### Deployment modes

The MovieLens licence forbids redistributing the ratings, so `data/raw/` is
gitignored and **is not present in the deployed container**. The API detects this
at boot and reports it as `has_interactions` on `/health`:

| `has_interactions` | When | Behaviour |
|---|---|---|
| `true` | Local, after `download_data.py` | Both paths; `user_id` works |
| `false` | Deployed | Cold start only; `user_id` returns `409` |

This is why the demo asks you to pick five films rather than logging you in.

### Semantic search

`GET /search` answers a free-text query — "space movies that are actually about
grief" — by encoding it into the same 384-d MiniLM space the documents live in
and ranking films by cosine. It is distinct from `GET /movies`, which matches
title substrings for the seed picker: a query whose words appear in no title
returns nothing from `/movies` and real results from `/search`.

Encoding a query at request time normally means shipping an ML runtime, which
this image does not have. `api/encoder.py` reimplements the MiniLM forward pass
— WordPiece tokenisation, six transformer blocks, mean pooling, L2 normalise —
in numpy and scipy, both of which are already serving dependencies.
`scripts/export_encoder.py` writes the tensors it reads as float16.

**Semantic search therefore adds no dependency to the serving image.** The
alternatives were `torch` (514 MB) or ONNX Runtime (124 MB with its sympy and
tokenizers chain, plus a 90 MB fp32 graph); this costs 41 MB of committed
artifact and nothing else. §14 records the reasoning.

Measured against `sentence-transformers` 6.0.0 on 304 real film overviews and
queries, worst-case cosine agreement is **0.999999**, and the exported
tokenizer reproduces HuggingFace's `input_ids` exactly on 2,000 overviews. The
regression test that guards this needs neither library: it re-encodes
`documents_sample.json` and asserts the vectors still match the rows those
documents produced in `content_emb.npy`.

Costs, measured: 41 MB on disk, 101 MB resident once loaded, 0.5 s to load, and
p50 37 ms per query with a p90 of 137 ms over 48 requests (the model runs on
4–20 tokens, so BLAS call overhead dominates, not arithmetic — which is also
why the tail is wide relative to the median). Loading is deferred to the first
`/search` call
rather than paid at boot, because 101 MB is a real fraction of a 512 MB free
tier and most sessions never search. `/health` reports `has_semantic_search`,
and `/search` returns `503` if the encoder artifacts are absent.

What it does *not* fix is compositional queries — see [F4](#12-test-cases),
which is now measured rather than assumed.

Title search still backs the seed picker, where matching what the user is
literally typing is the correct behaviour. The two are separate calls in the
frontend — `api.titleSearch` for the picker, `api.semanticSearch` for the
search box — and the search box submits on enter rather than debouncing per
keystroke, since each query is a real forward pass rather than a pandas mask.
The box is hidden entirely when `has_semantic_search` is false, so a
deployment without the encoder degrades to the rest of the product rather than
to a control that only returns 503.

---

## 10. Evaluation Methodology

### Split protocol
**Temporal, per user.** Each user's interactions are sorted by timestamp; the earliest 80% form the training set and the most recent 20% the test set. A random split would let the model observe a user's future when predicting their past, inflating every metric. We report only the temporal numbers.

Users with fewer than 5 interactions are excluded from evaluation (they cannot form a meaningful test set) but remain in training. In practice this filter never fires: GroupLens publishes only users with **at least 20 ratings**, so the dataset contains no genuinely cold users. Cold-start quality therefore cannot be measured directly — it is *simulated* by truncating a warm user history to its first *n* items. We report it as such rather than implying we observed real new users.

### Metrics

| Metric | Definition | What it tells us |
|---|---|---|
| **Precision@10** | Relevant items in top 10 ÷ 10 | Precision of the visible slate |
| **Recall@10** | Relevant items retrieved ÷ all relevant | Coverage of user interest |
| **NDCG@10** | Position-discounted gain | Whether good items rank *high*, not just present |
| **MAP@10** | Mean average precision | Ranking quality across the full list |
| **Coverage** | % of catalogue appearing in any top-10 | Long-tail reach |
| **Novelty** | mean `−log₂(popularity)` | Are we recommending the obvious? |
| **Intra-list diversity** | `1 − mean pairwise cosine` | Slate variety |
| **Latency p50 / p95** | End-to-end request time | Production viability |

Relevance threshold: a test item counts as relevant if rated **≥ 4.0**.

### Why these
Accuracy metrics alone reward recommending the ten most popular films to everyone — a system that scores well and serves users badly. Coverage, novelty, and diversity are reported alongside so that the accuracy/serendipity tradeoff is visible rather than hidden.

---

## 11. Results

All figures on the held-out temporal test split, k = 10.

| Model | P@10 | R@10 | NDCG@10 | MAP@10 | Coverage | Novelty | ILD | p95 (ms) |
|---|---|---|---|---|---|---|---|---|
| Random | 0.0017 | 0.0010 | 0.0015 | 0.0004 | 46.0% | 14.77 | 0.786 | 4 |
| Popularity | 0.0573 | 0.0498 | 0.0745 | 0.0356 | 1.0% | 8.57 | 0.718 | 3 |
| Item-kNN | 0.0400 | 0.0421 | 0.0572 | 0.0266 | 12.5% | 12.08 | 0.743 | 27 |
| ALS | 0.0730 | 0.0822 | 0.0993 | 0.0461 | 8.8% | 10.06 | 0.756 | 25 |
| Content-only | 0.0044 | 0.0042 | 0.0049 | 0.0015 | 6.2% | 14.86 | 0.534 | 27 |
| Hybrid | 0.0851 | 0.0837 | **0.1146** | **0.0565** | 7.3% | 9.83 | 0.711 | 26 |
| **Hybrid + MMR** (shipped) | 0.0848 | 0.0846 | 0.1144 | 0.0559 | 8.1% | 9.87 | **0.747** | 62 |

592 test users. Coverage is the share of the 9,742-item catalogue that appears in any user's top 10; novelty is mean negative log popularity; ILD is mean intra-list distance in content-embedding space.

**Reading the table.** The hybrid beats every single-model baseline on ranking: NDCG@10 0.1146 against 0.0993 for ALS, its strongest component, and 0.0745 for the popularity baseline. That margin over popularity is the one that matters — on a 610-user dataset with a temporal split, "recommend what everyone else watched" is a genuinely hard baseline, and a hybrid that failed to clear it would not be worth serving.

Two components look bad alone and still earn their place. Item-kNN (0.0572) underperforms popularity outright, and Content-only (0.0049) is barely above random. Content-only also has the *lowest* diversity in the table (ILD 0.534) — plot synopses cluster films by subject matter, so a content-ranked slate collapses into ten variations of one theme. But the content tower is what makes cold start possible at all: with no interaction history, ALS and item-CF have nothing to score against, and the 0.30 weight the grid search chose for it says the signal is real once it is blended rather than ranked on.

MMR costs almost nothing here. NDCG falls 0.1146 → 0.1144 (0.2%) and MAP 0.0565 → 0.0559, while ILD rises 0.711 → 0.747 and coverage 7.3% → 8.1%. That is a better trade than the validation sweep predicted, and it is why λ = 0.6 ships: a slate of ten near-identical films is a worse product than a marginally less "accurate" varied one, and at this magnitude the accuracy difference is within noise on 592 users.

The honest caveats: MMR more than doubles p95 latency (26 ms → 62 ms) because reranking is O(candidates × k) over 384-dimensional embeddings, and **coverage is poor across the board** — the shipped model surfaces 8.1% of the catalogue, so roughly nine films in ten are never recommended to anyone. Popularity's 1.0% shows where that pressure comes from. Fixing it needs an explicit long-tail term in the objective, not a λ adjustment.

Per-cohort NDCG@10, bucketed by training-history length: light (< 25 ratings, n=104) 0.1014 · medium (25–99, n=283) 0.1023 · heavy (≥ 100, n=205) 0.1378. The system degrades gracefully rather than collapsing on thin histories — light users retain 74% of heavy-user quality. Note that MovieLens guarantees at least 20 ratings per user, so "light" means light *for this dataset*; **there are no genuinely cold users in this measurement**, and the cold-start path the UI actually exercises is therefore untested offline.

Full breakdown, including per-cohort results (cold / warm / heavy users), in [`docs/EVALUATION.md`](docs/EVALUATION.md).

---

## 12. Test Cases

Detailed walkthroughs with screenshots in [`docs/TEST_CASES.md`](docs/TEST_CASES.md).

### ✅ Successful scenarios

| # | Scenario | Input | Expected behaviour | Result |
|---|---|---|---|---|
| S1 | Warm user, coherent taste | User 553 — 65 liked films, 43% Crime/Thriller | Genre-consistent recs with 2–3 adjacent-genre surprises | ✅ 8 of 10 are Crime/Drama/Thriller (American History X, Memento, Se7en, Boondock Saints, Shawshank); the surprises are Ferris Bueller's Day Off and Amélie at ranks 9–10. 27 ms |
| S2 | Cold start from 5 seeds | Toy Story · Pulp Fiction · The Matrix · Shawshank · Aliens | Sensible recs without any interaction history | ✅ Fight Club, Raiders of the Lost Ark, The Usual Suspects, Reservoir Dogs, Empire Strikes Back. `cold_start: true`, `als` contribution exactly `0.0` on every row, 7 ms |
| S3 | Item-to-item similarity | `/movies/2571/similar` (The Matrix) | Thematic neighbours, not just same-franchise | ⚠️ Passes but imperfectly. Top hits are thematic — Fight Club (0.391), Gattaca (0.340), Blade-Runner-adjacent sci-fi — yet 3 of 8 slots are Star Wars entries. That is genuine co-viewing signal, not a title-match artifact, but it reads as franchise clustering to a user |
| S4 | Natural-language search | `"mind-bending sci-fi"` | Semantically relevant results | ✅ Brainstorm (0.495), Mind Game, Paradox, Inception at rank 5. 22 ms. Partly lexical though — *A Beautiful Mind* rides the word "mind", not the concept. `"dystopian future ruled by machines"` is the cleaner win: The Terminator (0.491), The Time Machine (0.490), Terminator Genisys (0.479), Runaway, Equilibrium, Automata, Idiocracy — a list substring search returns **nothing** for, though *The Time Machine* does ride the literal "machines" |
| S5 | Explanation fidelity | User 553, all 10 recs | Named seed item is genuinely in user history; components sum to 1 | ✅ Both verified programmatically across the slate. Rank 1: American History X — `als 0.455, item_cf 0.225, content 0.227, popularity 0.093`, *because you watched Fight Club* (confirmed present in the user's liked set) |
| S6 | Eclectic user | User 115 — 73 liked films spanning 14 genres | MMR keeps the slate varied rather than collapsing to one genre | ✅ 13 distinct genres at λ=0.6 vs 10 at λ=1.0. The slate runs Goodfellas → Return of the Jedi → Blade Runner → Shrek → Rocky without losing coherence |

### ❌ Failure scenarios

| # | Scenario | What happens | Why | Mitigation status |
|---|---|---|---|---|
| F1 | **Extreme cold start** (1 seed) | Slate covers only one facet of taste. It does *not* collapse toward popularity — measured, popularity's share is **lower** at 1 seed (0.183) than at 5 (0.204) | One embedding is not a taste profile | Partially mitigated — UI labels it non-personalised |
| F2 | **Franchise degeneracy** | *The Godfather* → *The Godfather Part II* | Technically correct, practically useless: co-occurrence is near-perfect for sequels | Open |
| F3 | **Popularity bias** | Long-tail films rarely surface | ALS is trained on skewed interaction counts | Partially mitigated by MMR + novelty term |
| F4 | **Compositional queries** | *"like Inception but funny"* → *Inception* itself, then unrelated comedies (*It Came from Hollywood*, *The Emoji Movie*) | Measured: the modifier is **not** ignored — 7 of the top 8 are comedies — but the two concepts are averaged into a midpoint rather than intersected, so nothing returned is both | Open — needs query decomposition |
| F5 | **Stale catalogue** | Nothing after 2018 | Dataset boundary, not a model defect | Not fixable without new data |
| F6 | **Grey-sheep users** | Users with idiosyncratic taste get poor recs | Too few similar users for CF to work | Open — content weight helps marginally |

We consider F2 and F4 the most interesting: both are cases where the *metrics look fine* and the user experience is bad, which is exactly why offline evaluation alone is insufficient.

F1 is a case of the reverse — a failure mode we asserted from intuition and that
measurement contradicted. The original claim here was that a single seed
collapses the output toward popularity; running it showed the opposite, and the
row above now states what the numbers actually say.

F4 has now had the same treatment, since shipping `/search` finally made it
testable. The failure is real, but the stated *cause* was wrong: we claimed the
embedding anchors on *Inception* and ignores "funny". It does not ignore it —
almost everything returned is a comedy. What it cannot do is intersect the two
constraints, so it retrieves around the midpoint of "Inception" and "funny" and
returns films that are one or the other. That is a sharper argument for query
decomposition than the original guess was.

F6 remains not isolated by the current cohorts at all: history *length* is not
taste *idiosyncrasy*, and it is marked as such in
[`docs/TEST_CASES.md`](docs/TEST_CASES.md) rather than presented as measured.

---

## 13. Assumptions

1. A rating **≥ 4.0** signals genuine positive preference; below that is treated as neutral/negative.
2. Absence of a rating means *unseen*, not *disliked* — standard for implicit feedback, and it is why we use confidence weighting rather than treating zeros as negatives.
3. User taste is **stationary** within the dataset window. Over a 22-year span this is questionable; see [Limitations](#15-known-limitations).
4. MovieLens users are representative enough of general film audiences for the approach to generalise. They are not — MovieLens skews toward engaged, English-speaking, film-literate users.
5. Rating behaviour is honest and non-adversarial; no shilling or bot detection is implemented.
6. Watching a film is the conversion event we optimise for. A production system would optimise completion rate or long-term retention instead.

---

## 14. Key Design Decisions

| Decision | Alternatives considered | Rationale |
|---|---|---|
| **Hybrid over single model** | Pure ALS; pure content | Neither covers both cold and warm users. Hybrid degrades gracefully as history shrinks |
| **Precompute artifacts offline** | Train or compute on request | Keeps p95 latency in the tens of milliseconds and the container small enough for free-tier hosting |
| **No ML runtime in the serving image at all** | Ship `torch` + `sentence-transformers`; ship ONNX Runtime | Document embeddings are precomputed, so only the *query* needs encoding at request time — six transformer blocks over a handful of tokens. Neither runtime is proportionate to that: torch costs 519 MB (measured: torch 514, sentence-transformers 5), ONNX Runtime 124 MB (36 itself, sympy 69, plus the tokenizers/huggingface-hub chain) and a 90 MB fp32 graph. Reimplementing the forward pass in numpy costs 41 MB of float16 weights and no new dependency, and agrees with `sentence-transformers` to a cosine of 0.999999 (§9). Serving dependencies stay at 372 MB installed, dominated by scipy, pandas and pyarrow |
| **Top-50 neighbour truncation** | Full similarity matrix | 247× smaller artifact: 380 MB dense → 1.5 MB. The NDCG cost is unquantified — the full-matrix variant was never evaluated |
| **MMR reranking, shipped despite an accuracy cost** | Ship raw relevance ranking | A slate of ten near-identical films is a worse product than a slightly less "accurate" varied one |
| **Temporal split** | Random 80/20 | Random splits leak future interactions and inflate results. We preferred defensible numbers to flattering ones |
| **Explanations as first-class API output** | Post-hoc explanation generation | Contributions are computed during scoring, so the explanation reflects the *actual* decision rather than a plausible story told afterward |
| **React over Streamlit** | Streamlit single-app | Required for a credible Netflix-style rail layout, and demonstrates frontend/backend separation |

---

## 15. Known Limitations

- **No true online evaluation.** Every number here is offline. Without A/B testing, click-through and completion data, we cannot claim real user satisfaction — only ranking quality against historical behaviour.
- **Static model.** Artifacts are rebuilt manually; there is no incremental or streaming update. New ratings do not affect recommendations until a rebuild.
- **Small dataset.** 610 users is far below the scale where collaborative filtering shines; results should not be extrapolated to production scale.
- **Popularity bias persists** despite MMR — see F3.
- **No sequential/temporal modelling.** We ignore the *order* of a user's viewing history, which discards real signal a session-based or transformer model would exploit.
- **No fairness or provider-side analysis.** We do not measure whether the system disadvantages particular genres, eras, or regions.
- **Explanations are faithful but shallow.** They report which component dominated, not a causal account of *why* that component fired.
- **Semantic search retrieves on plot, not craft.** 97.4% of documents carry a TMDB overview, which describes what happens in a film — so a query about story works, and one about cinematography, pacing or tone has almost nothing to match against. Queries also partly match on literal words: *"mind-bending sci-fi"* surfaces *A Beautiful Mind*.
- **Search cannot compose constraints** — see F4. It embeds a whole query as one point, so "like X but Y" retrieves the midpoint of X and Y rather than their intersection.
- **Single-region deployment**, no CDN for the API, no horizontal scaling.
- **Cold start for new *items*** is unhandled — a film with no ratings can only ever be reached through content similarity.

---

## 16. Comparison with Netflix

The UI borrows two things from Netflix's browse experience: a horizontally
scrolling rail of recommendations, and explanations phrased as *"because you
watched…"*. It is not a clone of that layout — there is no hero banner, and
there is one rail, not a page of genre rows. The differences below are the
substantive comparison; this section is about architecture, not chrome.

### Similarities
- **Horizontally scrolling rail** for the recommendation slate. Netflix's browse page is many such rails selected per user; CineMatch renders one. Per-row reason titles are the mechanism we did *not* build — the explanation lives per-card in a drawer instead (see the personalisation-surface row below).
- **Hybrid retrieval-then-rank pipeline** — candidate generation followed by a scoring pass, the same broad shape as Netflix's production architecture.
- **Diversity as an explicit objective**, not an accident of ranking.
- **Cold-start onboarding** via a taste picker, mirroring Netflix's original signup flow.

### Differences

| | Netflix | CineMatch |
|---|---|---|
| Signal | Completion rate, dwell, session context, device, time of day | Explicit star ratings only |
| Scale | ~300M members, continuous retraining | 610 users, manual rebuild |
| Personalisation surface | Row *selection*, row *order*, and artwork per user | Row content only |
| Ranking model | Deep learning ensembles, bandits, contextual models | ALS + kNN + embeddings |
| Evaluation | Continuous online A/B at scale | Offline temporal split |
| Catalogue | Licensed, dynamic, with business constraints | Static, to 2018 |

### Current limitations relative to Netflix
No artwork personalisation, no contextual or session awareness, no exploration/exploitation strategy, no business-rule layer (licensing windows, promotional pushes), and no feedback loop from what the user actually did next.

### What we would build next with more time
1. **Session-based sequential model** (SASRec/BERT4Rec) to capture short-term intent — the single largest expected quality gain.
2. **Contextual bandit for exploration**, so the system learns from what it shows rather than only from what it has already seen.
3. **Personalised row selection and ordering**, which is where much of Netflix's real personalisation lives.
4. **Online evaluation harness** — interleaving or A/B, with CTR and completion as primary metrics.
5. **Two-tower retrieval with an ANN index** to scale candidate generation past a small catalogue.

---

## 17. Future Improvements

**Modelling** — sequential/session-aware recommenders; two-tower retrieval with FAISS; learning-to-rank reranker (LambdaMART or a small neural ranker); query decomposition for compositional natural-language search, splitting "like X but Y" into constraints that are intersected rather than averaged (addressing the measured F4); franchise-collapse penalty (addressing F2).

**Engineering** — incremental artifact rebuilds on a schedule; Redis caching for hot users; model registry and versioned artifacts; proper observability (per-endpoint latency, recommendation distribution drift); CI running the evaluation suite so metric regressions block merges.

**Product** — thumbs up/down feedback loop; "less like this" negative signal; explanation depth toggle; watchlist persistence; per-row diversity controls exposed to the user.

**Evaluation** — per-cohort dashboards; fairness analysis across genre/era/region; counterfactual offline evaluation (IPS) to better approximate online performance.

---

## 18. Licensing & Attribution

- **MovieLens data** © GroupLens Research, University of Minnesota. Used under the [MovieLens terms of use](https://files.grouplens.org/datasets/movielens/ml-latest-small-README.html). **Redistribution is not permitted** — raw data is excluded from this repository and fetched at setup time.
  > F. M. Harper and J. A. Konstan. 2015. *The MovieLens Datasets: History and Context.* ACM TiiS 5, 4.
- **Poster imagery** courtesy of [The Movie Database (TMDB)](https://www.themoviedb.org/). This product uses the TMDB API but is not endorsed or certified by TMDB.
- **Netflix** is a trademark of Netflix, Inc. This project is an independent educational exercise with no affiliation to or endorsement by Netflix.
- Project code released under the MIT Licence — see [`LICENSE`](LICENSE).

---

*Built as a technical assignment submission by ⟨TODO: your name⟩ · ⟨TODO: date⟩*
