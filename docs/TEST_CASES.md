# Test Cases

Every result below was produced by running the scenario against the live API
on the committed artifacts, not written from expectation. Reproduce with the
`curl` commands shown, or run `pytest` for the automated subset.

Configuration under test: weights `als 0.4 · item_cf 0.2 · content 0.3 ·
popularity 0.1`, λ = 0.6, k = 10, artifacts built 2026-08-19.

**Summary**

| # | Scenario | Result |
|---|---|---|
| [S1](#s1--warm-user-coherent-taste) | Warm user, coherent taste | Pass |
| [S2](#s2--cold-start-from-five-seeds) | Cold start from 5 seeds | Pass |
| [S3](#s3--item-to-item-similarity) | Item-to-item similarity | Pass with a caveat |
| [S4](#s4--natural-language-search) | Natural-language search | Not shipped |
| [S5](#s5--explanation-fidelity) | Explanation fidelity | Pass |
| [S6](#s6--eclectic-user) | Eclectic user, MMR effect | Pass |
| [F1](#f1--extreme-cold-start-one-seed) | Extreme cold start | **README claim was wrong — corrected** |
| [F2](#f2--franchise-degeneracy) | Franchise degeneracy | Confirmed, open |
| [F3](#f3--popularity-bias-and-coverage) | Popularity bias | Confirmed, partially mitigated |
| [F4](#f4--compositional-queries) | Compositional queries | Untestable — depends on S4 |
| [F5](#f5--stale-catalogue) | Stale catalogue | Confirmed, not fixable |
| [F6](#f6--grey-sheep-users) | Grey-sheep users | Not isolated by the current split |

---

## Successful scenarios

### S1 — Warm user, coherent taste

**Setup.** User 553, selected as the most genre-focused user with a substantial
history: 65 liked films, genre counts `Drama 42 · Crime 41 · Thriller 29 ·
Action 15 · Comedy 11`.

```bash
curl -s -X POST localhost:8000/recommend \
  -H 'Content-Type: application/json' -d '{"user_id": 553, "k": 10}'
```

**Result — 37.2 ms, `cold_start: false`, `personalised: true`**

| # | Film | Genres | als | item_cf | content | pop | Because you watched |
|---|---|---|---|---|---|---|---|
| 1 | American History X | Crime, Drama | 0.455 | 0.225 | 0.227 | 0.093 | Fight Club (0.416) |
| 2 | Donnie Darko | Drama, Mystery, Sci-Fi | 0.461 | 0.120 | 0.291 | 0.128 | Eternal Sunshine (0.418) |
| 3 | Memento | Mystery, Thriller | 0.428 | 0.181 | 0.275 | 0.116 | Fight Club (0.432) |
| 4 | The Boondock Saints | Action, Crime, Drama | 0.360 | 0.273 | 0.285 | 0.083 | The Departed (0.468) |
| 5 | The Shawshank Redemption | Crime, Drama | 0.427 | 0.133 | 0.296 | 0.143 | The Usual Suspects (0.420) |
| 6 | Se7en | Mystery, Thriller | 0.405 | 0.163 | 0.291 | 0.141 | Fight Club (0.380) |
| 7 | Platoon | Drama, War | 0.358 | 0.178 | 0.347 | 0.116 | Goodfellas (0.408) |
| 8 | Blow | Crime, Drama | 0.486 | 0.016 | 0.388 | 0.110 | The Departed (0.281) |
| 9 | Ferris Bueller's Day Off | Comedy | 0.541 | 0.043 | 0.280 | 0.137 | Office Space (0.266) |
| 10 | Amélie | Comedy, Romance | 0.425 | 0.104 | 0.323 | 0.147 | Requiem for a Dream (0.331) |

**Verdict: pass.** 8 of 10 carry Crime, Thriller or Drama, matching the
history. Ranks 9–10 are the adjacent-genre surprises the scenario asks for —
and note they are *not* random noise: Ferris Bueller is attributed to Office
Space and Amélie to Requiem for a Dream, both genuinely in the user's history.
The ALS term dominates throughout (0.36–0.54), which is what a warm user with
65 ratings should look like.

---

### S2 — Cold start from five seeds

**Setup.** No user, five seeds: Toy Story (1), Pulp Fiction (296), The Matrix
(2571), The Shawshank Redemption (318), Aliens (1200). Deliberately incoherent
— a children's film, two crime films, a prison drama and a sci-fi horror — to
check the system does not simply mirror one of them.

```bash
curl -s -X POST localhost:8000/recommend -H 'Content-Type: application/json' \
  -d '{"seed_movie_ids": [1, 296, 2571, 318, 1200], "k": 10}'
```

**Result — 9.7 ms, `cold_start: true`, `personalised: true`**

| # | Film | als | item_cf | content | pop | Because you watched |
|---|---|---|---|---|---|---|
| 1 | Fight Club | **0.000** | 0.328 | 0.485 | 0.187 | Pulp Fiction (0.456) |
| 2 | Raiders of the Lost Ark | **0.000** | 0.367 | 0.441 | 0.192 | The Matrix (0.313) |
| 3 | The Usual Suspects | **0.000** | 0.357 | 0.434 | 0.208 | Pulp Fiction (0.459) |
| 4 | Reservoir Dogs | **0.000** | 0.386 | 0.440 | 0.175 | Pulp Fiction (0.405) |
| 5 | Saving Private Ryan | **0.000** | 0.383 | 0.403 | 0.214 | The Matrix (0.371) |
| 6 | The Empire Strikes Back | **0.000** | 0.444 | 0.351 | 0.205 | The Matrix (0.386) |
| 7 | Goodfellas | **0.000** | 0.375 | 0.434 | 0.191 | Pulp Fiction (0.379) |
| 8 | The Silence of the Lambs | **0.000** | 0.384 | 0.370 | 0.246 | Pulp Fiction (0.390) |
| 9 | Office Space | **0.000** | 0.257 | 0.553 | 0.190 | The Matrix (0.287) |
| 10 | Forrest Gump | **0.000** | 0.199 | 0.568 | 0.234 | Shawshank (0.440) |

**Verdict: pass.** Asserted programmatically: every `als` contribution is
exactly `0.0` (no latent factors exist for `user = -1`), and no seed is
recommended back. Content carries the largest share on 8 of 10 rows, which is
the intended division of labour when collaborative signal is unavailable.

The slate spreads across the seeds rather than collapsing onto one — four rows
trace to Pulp Fiction, four to The Matrix, one to Shawshank. Toy Story and
Aliens contribute nothing visible, which is the honest weakness of this case:
the animated and horror ends of the seed set are outvoted.

---

### S3 — Item-to-item similarity

```bash
curl -s 'localhost:8000/movies/2571/similar?k=8'
```

**The Matrix (2571)**

| cosine | Film |
|---|---|
| 0.391 | Fight Club |
| 0.386 | Star Wars: Episode V — The Empire Strikes Back |
| 0.371 | Saving Private Ryan |
| 0.343 | Die Hard |
| 0.340 | Gattaca |
| 0.335 | Reservoir Dogs |
| 0.333 | Star Wars: Episode VI — Return of the Jedi |
| 0.328 | Star Wars: Episode IV — A New Hope |

**Verdict: pass with a caveat.** The thematic neighbours are real — Gattaca and
Blade-Runner-adjacent sci-fi, Fight Club at the top — but **3 of 8 slots are
Star Wars entries**. This is genuine co-viewing signal rather than a
title-matching artifact, and the similarity values are unremarkable (0.33
against a 0.39 ceiling), but to a user it reads as franchise clustering. See
[F2](#f2--franchise-degeneracy).

---

### S4 — Natural-language search

**Not shipped.** `GET /search` does not exist. Answering a free-text query
requires encoding it at request time, which needs a MiniLM ONNX export that
has not been built, and a runtime to execute it that is deliberately not
installed. See README §9 and §14. Title search covers the seed picker, which
is what the UI needs.

---

### S5 — Explanation fidelity

Two properties, both checked across all 10 recommendations for user 553:

1. **Contributions sum to 1.0.** Measured range across the slate:
   `min 0.999999, max 1.000001` — float rounding at six decimal places, nothing
   more.
2. **The named seed is genuinely in the user's history.** True for every row,
   verified against the user's liked set rather than trusted.

**Verdict: pass.** This matters because the contributions are captured *during*
scoring rather than reconstructed afterwards, so the panel cannot tell a
plausible story that disagrees with the ranking. The same two properties are
asserted in `tests/test_api.py` so a regression fails the build.

---

### S6 — Eclectic user

**Setup.** User 115, the widest genre spread in the dataset: 73 liked films
across 14 genres, `Comedy 35 · Action 28 · Thriller 23 · Adventure 22 ·
Drama 17`.

| λ = 1.0 (relevance only) — **10 genres**, 30.3 ms | λ = 0.6 (shipped) — **13 genres**, 46.6 ms |
|---|---|
| Goodfellas | Goodfellas |
| One Flew Over the Cuckoo's Nest | Return of the Jedi |
| The Godfather | Die Hard |
| Die Hard | One Flew Over the Cuckoo's Nest |
| Blade Runner | Blade Runner |
| **The Godfather Part II** | **Shrek** |
| Return of the Jedi | **Rocky** |
| Dr. Strangelove | **Indiana Jones and the Last Crusade** |
| The Good, the Bad and the Ugly | **Glory** |
| Reservoir Dogs | Reservoir Dogs |

**Verdict: pass.** MMR displaces *The Godfather* and *Godfather Part II* — the
greedy slate carried both — and brings in Shrek, Rocky, Indiana Jones and
Glory. Genre count rises 10 → 13 and the slate stops reading as one mood. The
cost is latency: 30 ms → 47 ms, because reranking is O(candidates × k) over
384-dimensional embeddings.

---

## Failure scenarios

### F1 — Extreme cold start (one seed)

**The README's original claim was that output "collapses toward popularity"
with a single seed. That is not what happens.** Measured mean contribution
across the top 10, seeding from Pulp Fiction and adding seeds:

| Seeds | `personalised` | popularity | item_cf | content |
|---|---|---|---|---|
| 1 | false | **0.183** | 0.328 | 0.489 |
| 2 | false | 0.191 | 0.338 | 0.470 |
| 3 | true | 0.191 | 0.337 | 0.471 |
| 5 | true | **0.204** | 0.348 | 0.448 |

Popularity's share **rises** with more seeds, from 0.183 to 0.204 — the
opposite of the claim. Content's share falls correspondingly. The one-seed
slate is *more* content-driven, not less, and it is perfectly coherent:
Fight Club, Memento, The Usual Suspects, Reservoir Dogs, The Godfather — all
recognisably Pulp-Fiction-adjacent.

**The real failure is narrower than advertised.** One seed does not degrade
quality much on a well-connected film; it degrades *coverage of the user's
taste*, because a single embedding cannot express a taste that spans genres.
The UI still correctly reports `personalised: false` below three seeds.

**Status: README corrected.** The mitigation (labelling it non-personalised) is
right; the stated mechanism was not.

---

### F2 — Franchise degeneracy

**Confirmed, and it is the most visible weakness in the system.**

```
The Godfather      -> Godfather Part II (0.728) | Goodfellas (0.560) | Reservoir Dogs (0.416)
LOTR: Fellowship   -> The Two Towers (0.767)    | Return of the King (0.714) | Spirited Away (0.313)
```

Note the *gap*. For Fellowship, the two sequels score 0.767 and 0.714 while the
third-place neighbour drops to 0.313 — a cliff, not a gradient. Co-viewing
within a trilogy is near-perfect, so item-CF ranks sequels far above anything
thematically interesting.

This is technically correct and practically useless: a user who just watched
Fellowship does not need to be told The Two Towers exists. MMR mitigates it in
`/recommend` (see S6, where both Godfathers were displaced), but
`/movies/{id}/similar` returns raw neighbours with no reranking, so the drawer
in the UI shows it plainly.

**Status: open.** A franchise-aware penalty — down-weighting neighbours sharing
a title prefix or a collection id from TMDB — would fix the visible symptom.
Not implemented.

---

### F3 — Popularity bias and coverage

**Confirmed.** Catalogue coverage across the 9,742 films, from the measured
evaluation:

| Model | Coverage | Novelty |
|---|---|---|
| Random | 46.0% | 14.77 |
| Popularity | 1.0% | 8.57 |
| Item-kNN | 12.5% | 12.08 |
| ALS | 8.8% | 10.06 |
| Content-only | 6.2% | 14.86 |
| Hybrid | 7.3% | 9.83 |
| **Hybrid + MMR (shipped)** | **8.1%** | 9.87 |

The shipped model reaches 8.1% of the catalogue. **Roughly nine films in ten
are never recommended to anyone.** MMR helps — 7.3% → 8.1%, and novelty ticks
up 9.83 → 9.87 — but the effect is small next to the size of the problem.

**Status: partially mitigated.** Closing this needs an explicit long-tail term
in the objective, not a λ adjustment. λ trades diversity *within* a slate;
it does not push the system toward unpopular items.

---

### F4 — Compositional queries

**Untestable here.** This failure mode is a property of semantic search, which
is not shipped ([S4](#s4--natural-language-search)). Listing it as an observed
failure would be dishonest — the claim is that *"like Inception but funny"*
would anchor on Inception and ignore the modifier, which is the expected
behaviour of single-vector embedding retrieval, but it has not been measured on
this system.

**Status: open, unverified.**

---

### F5 — Stale catalogue

**Confirmed.** The catalogue spans to **2018**, with 680 of 9,742 films from
2015 onward. This is the MovieLens `ml-latest-small` boundary, not a model
defect, and nothing in the pipeline can fix it without a newer dataset.

A related data-quality note surfaced while testing: **13 films carry no usable
year** (stored as 0), because the year is parsed out of the MovieLens title
string and those titles lack one — *Babylon 5*, *Ready Player One*, *Hyena
Road*, *Nocturnal Animals* among them. The UI renders these as "—" rather than
"0", so the defect is contained, but the underlying values are wrong.

**Status: not fixable without new data.** The year-parsing gap is fixable and
is not currently tracked.

---

### F6 — Grey-sheep users

**Not isolated by the current evaluation.** Per-cohort NDCG@10, bucketed by
training-history length:

| Cohort | n | NDCG@10 |
|---|---|---|
| Light (< 25 ratings) | 104 | 0.1014 |
| Medium (25–99) | 283 | 0.1023 |
| Heavy (≥ 100) | 205 | 0.1378 |

Light users retain 74% of heavy-user quality, so the system degrades gracefully
rather than collapsing. But **this does not measure grey sheep.** History
*length* is not taste *idiosyncrasy* — a user with 200 ratings can still be a
grey sheep. Testing this properly needs a cohort defined by low mean similarity
to the rest of the user base, which the current harness does not compute.

**Status: open, and the existing metric does not address it.**

---

## Automated coverage

`tests/test_api.py` — 20 tests against the real committed artifacts, not
synthetic fixtures, so a broken artifact fails the suite rather than passing
against a fabricated one.

```bash
pytest
```

Asserted properties include: contributions sum to 1.0, seeds are never
recommended back, `als == 0.0` on the cold-start path, scores descend at
λ = 1.0, MMR widens genre spread, no duplicate recommendations, single-seed
requests are not labelled personalised, and the 404 / 422 / 409 error paths.
