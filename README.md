# CineMatch — A Hybrid, Explainable Movie Recommendation System

> **Live demo:** ⟨TODO: deployment URL⟩
> **API docs:** ⟨TODO: backend URL⟩/docs
> **Repository:** ⟨TODO: GitHub URL⟩

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

The core difficulty is sparsity. In our dataset the user–item matrix is **⟨TODO: X⟩% dense**, meaning the average user has rated a vanishingly small fraction of the catalog. A system that only memorizes co-occurrence patterns degenerates into recommending blockbusters to everyone.

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
│  SERVING  —  FastAPI on ⟨TODO: host⟩                     │
│                                                          │
│   GET /recommend?user_id=&k=                             │
│   GET /similar/{movie_id}                                │
│   POST /cold_start        (seed items → profile)         │
│   GET /search?q=          (natural-language query)       │
│   GET /explain/{user_id}/{movie_id}                      │
└──────────────────────────────────────────────────────────┘
                              │  JSON
                              ▼
┌──────────────────────────────────────────────────────────┐
│  FRONTEND  —  React + Vite on ⟨TODO: host⟩               │
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
│   └── evaluate.py
├── api/
│   ├── main.py            # FastAPI app
│   ├── recommender.py     # scoring + rerank + explain
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
Cosine similarity over the L2-normalised, mean-centred user–item matrix. We keep the **top 50 neighbours per item** and discard the rest, which cuts the similarity matrix from ⟨TODO⟩ MB to ⟨TODO⟩ MB with negligible metric loss.

### 5.2 Matrix factorisation (ALS)
Alternating Least Squares on the implicit-feedback formulation, with ratings ≥ 4.0 treated as positive signal and confidence weighting `c = 1 + α·r`.

- Factors: ⟨TODO⟩ · Regularisation: ⟨TODO⟩ · Iterations: ⟨TODO⟩ · α: ⟨TODO⟩
- Selected by grid search on the validation split (see `scripts/evaluate.py`).

### 5.3 Content embeddings
Each film is represented as a single text document — `title · genres · top user tags · overview` — encoded with `all-MiniLM-L6-v2` into a 384-dimensional vector. These power content similarity, cold-start profiles, natural-language search, and the MMR diversity term.

### 5.4 Hybrid scoring

```
score(u, i) = w₁·ALS(u,i) + w₂·itemCF(u,i) + w₃·content(u,i) + w₄·popularity(i)
```

Each component is min–max normalised within the candidate set before blending, so no single signal dominates by virtue of its scale. Weights: `w₁=⟨TODO⟩, w₂=⟨TODO⟩, w₃=⟨TODO⟩, w₄=⟨TODO⟩`, tuned on the validation split against NDCG@10.

### 5.5 Diversity reranking (MMR)

```
MMR(i) = λ · score(u,i) − (1−λ) · max_{j ∈ selected} cos(emb_i, emb_j)
```

with **λ = ⟨TODO⟩**. This deliberately trades a small amount of ranking accuracy for a large gain in intra-list diversity and catalogue coverage — quantified in [Results](#11-results).

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
| Density | ⟨TODO⟩% |

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
| Runtime inference | ONNX Runtime (quantised MiniLM) |
| API | FastAPI, Uvicorn, Pydantic v2 |
| Frontend | React 18, Vite, TypeScript, Tailwind CSS |
| Deployment | ⟨TODO: backend host⟩ (Docker), ⟨TODO: frontend host⟩ |
| Testing | pytest, ruff |

---

## 8. Getting Started

### Prerequisites
Python 3.11+, Node 18+, ~2 GB free disk.

### Backend

```bash
git clone ⟨TODO: repo URL⟩
cd recommendation_system

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python scripts/download_data.py     # fetches MovieLens into data/raw/
python scripts/build_artifacts.py   # ~⟨TODO⟩ min — trains and writes artifacts/
uvicorn api.main:app --reload       # http://localhost:8000/docs
```

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
```

---

## 9. API Reference

Interactive docs at `/docs`. Summary:

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/recommend?user_id={id}&k=10` | Personalised recommendations with explanations |
| `GET` | `/similar/{movie_id}?k=10` | Item-to-item "More like this" |
| `POST` | `/cold_start` | Body: `{ "movie_ids": [...] }` → recommendations from seeds |
| `GET` | `/search?q={query}` | Natural-language semantic search |
| `GET` | `/explain/{user_id}/{movie_id}` | Full score decomposition for one pair |
| `GET` | `/health` | Liveness + artifact load status |

<details>
<summary>Example response — <code>GET /recommend?user_id=42&k=2</code></summary>

```json
{
  "user_id": 42,
  "generated_in_ms": 31,
  "strategy": "hybrid+mmr",
  "recommendations": [
    {
      "movie_id": 1213,
      "title": "Goodfellas (1990)",
      "score": 0.891,
      "explanation": {
        "primary_reason": "Because you watched Heat (1995)",
        "components": { "als": 0.68, "item_cf": 0.24, "content": 0.06, "popularity": 0.02 }
      }
    }
  ]
}
```
</details>

---

## 10. Evaluation Methodology

### Split protocol
**Temporal, per user.** Each user's interactions are sorted by timestamp; the earliest 80% form the training set and the most recent 20% the test set. A random split would let the model observe a user's future when predicting their past, inflating every metric. We report only the temporal numbers.

Users with fewer than 5 interactions are excluded from evaluation (they cannot form a meaningful test set) but remain in training.

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
| Random | ⟨TODO⟩ | ⟨TODO⟩ | ⟨TODO⟩ | ⟨TODO⟩ | ⟨TODO⟩ | ⟨TODO⟩ | ⟨TODO⟩ | ⟨TODO⟩ |
| Popularity | ⟨TODO⟩ | ⟨TODO⟩ | ⟨TODO⟩ | ⟨TODO⟩ | ⟨TODO⟩ | ⟨TODO⟩ | ⟨TODO⟩ | ⟨TODO⟩ |
| Item-kNN | ⟨TODO⟩ | ⟨TODO⟩ | ⟨TODO⟩ | ⟨TODO⟩ | ⟨TODO⟩ | ⟨TODO⟩ | ⟨TODO⟩ | ⟨TODO⟩ |
| ALS | ⟨TODO⟩ | ⟨TODO⟩ | ⟨TODO⟩ | ⟨TODO⟩ | ⟨TODO⟩ | ⟨TODO⟩ | ⟨TODO⟩ | ⟨TODO⟩ |
| Content-only | ⟨TODO⟩ | ⟨TODO⟩ | ⟨TODO⟩ | ⟨TODO⟩ | ⟨TODO⟩ | ⟨TODO⟩ | ⟨TODO⟩ | ⟨TODO⟩ |
| Hybrid | ⟨TODO⟩ | ⟨TODO⟩ | ⟨TODO⟩ | ⟨TODO⟩ | ⟨TODO⟩ | ⟨TODO⟩ | ⟨TODO⟩ | ⟨TODO⟩ |
| **Hybrid + MMR** (shipped) | ⟨TODO⟩ | ⟨TODO⟩ | ⟨TODO⟩ | ⟨TODO⟩ | ⟨TODO⟩ | ⟨TODO⟩ | ⟨TODO⟩ | ⟨TODO⟩ |

**Reading the table:** ⟨TODO — write 3–5 sentences of honest interpretation. Expected shape: the hybrid beats every single-model baseline on ranking metrics; adding MMR *costs* a little NDCG while substantially raising coverage and intra-list diversity. State that tradeoff explicitly and justify shipping it.⟩

Full breakdown, including per-cohort results (cold / warm / heavy users), in [`docs/EVALUATION.md`](docs/EVALUATION.md).

---

## 12. Test Cases

Detailed walkthroughs with screenshots in [`docs/TEST_CASES.md`](docs/TEST_CASES.md).

### ✅ Successful scenarios

| # | Scenario | Input | Expected behaviour | Result |
|---|---|---|---|---|
| S1 | Warm user, coherent taste | User ⟨TODO⟩ (crime/thriller history) | Genre-consistent recs with 2–3 adjacent-genre surprises | ⟨TODO⟩ |
| S2 | Cold start from 5 seeds | Seeds: ⟨TODO⟩ | Sensible recs without any interaction history | ⟨TODO⟩ |
| S3 | Item-to-item similarity | `/similar/{id}` | Thematic neighbours, not just same-franchise | ⟨TODO⟩ |
| S4 | Natural-language search | `"mind-bending sci-fi"` | Semantically relevant results | ⟨TODO⟩ |
| S5 | Explanation fidelity | Any rec | Named seed item is genuinely in user history; components sum to 1 | ⟨TODO⟩ |
| S6 | Eclectic user | User with wide genre spread | MMR keeps the slate varied rather than collapsing to one genre | ⟨TODO⟩ |

### ❌ Failure scenarios

| # | Scenario | What happens | Why | Mitigation status |
|---|---|---|---|---|
| F1 | **Extreme cold start** (1 seed) | Output collapses toward popularity | One embedding is not a taste profile | Partially mitigated — UI labels it non-personalised |
| F2 | **Franchise degeneracy** | *The Godfather* → *The Godfather Part II* | Technically correct, practically useless: co-occurrence is near-perfect for sequels | Open |
| F3 | **Popularity bias** | Long-tail films rarely surface | ALS is trained on skewed interaction counts | Partially mitigated by MMR + novelty term |
| F4 | **Compositional queries** | *"like Inception but funny"* | Embedding anchors on *Inception*, ignores the negation/modifier | Open — needs query decomposition |
| F5 | **Stale catalogue** | Nothing after 2018 | Dataset boundary, not a model defect | Not fixable without new data |
| F6 | **Grey-sheep users** | Users with idiosyncratic taste get poor recs | Too few similar users for CF to work | Open — content weight helps marginally |

We consider F2 and F4 the most interesting: both are cases where the *metrics look fine* and the user experience is bad, which is exactly why offline evaluation alone is insufficient.

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
| **ONNX Runtime instead of PyTorch at serving time** | Ship `torch` + `sentence-transformers` | Reduces the image by ~⟨TODO⟩ MB. Serving needs inference only, never gradients |
| **Top-50 neighbour truncation** | Full similarity matrix | ⟨TODO⟩× smaller artifact for ⟨TODO⟩ NDCG loss |
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
- **Single-region deployment**, no CDN for the API, no horizontal scaling.
- **Cold start for new *items*** is unhandled — a film with no ratings can only ever be reached through content similarity.

---

## 16. Comparison with Netflix

The UI is deliberately modelled on Netflix's browse experience: a hero banner, horizontally scrolling genre rails, and reason-labelled rows.

### Similarities
- **Rail-based browse layout** with reason-carrying row titles (*"Because you watched…"*), which is Netflix's primary explanation mechanism.
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

**Modelling** — sequential/session-aware recommenders; two-tower retrieval with FAISS; learning-to-rank reranker (LambdaMART or a small neural ranker); query decomposition for compositional natural-language search (addressing F4); franchise-collapse penalty (addressing F2).

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
