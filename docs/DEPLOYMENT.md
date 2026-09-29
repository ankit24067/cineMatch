# Deployment

Two services, deployed separately: a Dockerised FastAPI backend and a static
frontend build. Both run on free tiers.

Everything the API serves is committed to the repository, so there is no build
step, no database, and no external service to provision. `artifacts/` is 68 MB
in git; the image carries about 64 MB of it (see [What ships](#what-ships)),
of which the query encoder is 42 MB.

---

## The one thing to understand first

**The deployed API has no interaction data.** The MovieLens licence forbids
redistributing the raw ratings, so `data/raw/` is gitignored and therefore
absent from the image. `api/artifacts.py` detects this at boot and degrades
deliberately rather than crashing:

| | Local (with `data/raw/`) | Deployed |
|---|---|---|
| `/health` → `has_interactions` | `true` | `false` |
| Cold start from `seed_movie_ids` | works | works |
| `/recommend` with `user_id` | works | **409** with an explanatory message |
| `/movies`, `/movies/{id}/similar`, `/evaluation` | work | work |

The frontend only ever uses the cold-start path, so it is fully functional
against a deployed API. Nothing in the UI is degraded.

---

## Backend — Render (Docker)

1. Push this repository to GitHub.
2. Render dashboard → **New** → **Blueprint** → select the repo. Render reads
   [`render.yaml`](../render.yaml) and creates both services.
3. Wait for the first build. It installs `requirements.txt` and copies the
   artifacts; expect roughly 3–5 minutes.
4. Confirm it is live:

   ```bash
   curl https://<your-api>.onrender.com/health
   ```

   You want `"status": "ok"`, `"n_items": 9742`, and
   `"has_interactions": false`.

To deploy the backend alone instead of the blueprint: **New → Web Service**,
runtime **Docker**, health check path `/health`. No environment variables are
required — the Dockerfile's `CMD` already honours the `PORT` that Render
injects.

### Free-tier behaviour worth knowing

- **The instance sleeps after ~15 minutes idle.** The next request pays a cold
  start of roughly 30–60 seconds while the container boots and loads artifacts.
  This is the single worst thing about the free tier for a demo you are asking
  someone to click. If that matters, either upgrade the instance or warm it
  with an external ping before the demo.
- **Memory.** Free instances get 512 MB. The recommender artifacts are ~22 MB of
  numpy held resident from boot, and numpy/scipy/pandas dominate the rest. The
  query encoder is the one large addition: it loads lazily on the first
  `/search` and then holds ~101 MB for the life of the process, so a container
  that has served a search sits far higher than one that has not. It fits, but
  the Dockerfile runs a **single** uvicorn worker on purpose — a second worker
  duplicates both the artifacts and the encoder for no throughput gain at this
  scale.

---

## Frontend — Render static site, or Vercel

The blueprint already defines the static site. If you deploy it separately
instead, the settings are:

| Setting | Value |
|---|---|
| Root directory | `frontend` |
| Build command | `npm ci && npm run build` |
| Publish directory | `dist` |
| SPA rewrite | `/*` → `/index.html` |

**`VITE_API_URL` is inlined at build time, not read at runtime.** Vite
substitutes it into the bundle during `npm run build`, so changing it requires
a *rebuild*, not a restart. If the frontend is pointing at the wrong API, a
redeploy is the fix — restarting will change nothing.

Set it to the backend's full origin with no trailing slash:

```
VITE_API_URL=https://<your-api>.onrender.com
```

For **Vercel**, [`frontend/vercel.json`](../frontend/vercel.json) carries the
same configuration. Import the repo, set the root directory to `frontend`, and
add `VITE_API_URL` as an environment variable. Netlify works identically with
`base = "frontend"` and `publish = "dist"`.

---

## What ships

The image contains only what the API reads at request time:

| Included | Why |
|---|---|
| `api/` | The application |
| `artifacts/` minus two files | `catalogue.parquet`, `content_emb.npy`, `item_factors.npy`, `user_factors.npy`, `neighbors_top50.npz`, `popularity.parquet`, `id_maps.json`, `manifest.json`, `evaluation.json`, `tuned_weights.json` (~22 MB) |
| `artifacts/encoder/` | `weights.npz`, `vocab.txt`, `config.json` (42 MB) — the MiniLM tensors `api/encoder.py` reads to encode a `/search` query. Omit them and every other endpoint still serves; see below |
| `scripts/` | Imported lazily by `api/artifacts.py`, and only when `data/raw` is mounted. Small, and copying it keeps that path working |

Excluded by [`.dockerignore`](../.dockerignore):

- **`artifacts/tmdb.json`** (4.1 MB) — the raw TMDB response cache consumed by
  `build_artifacts.py`. The catalogue already carries the posters and
  overviews, and nothing at serving time opens it.
- `artifacts/documents_sample.json`, `frontend/`, `notebooks/`, `tests/`,
  `docs/`, `data/`, `.venv/`, `.git/`.

No ML runtime is installed. `onnxruntime` and `tokenizers` stay commented out of
`requirements.txt`: semantic search encodes its queries with `api/encoder.py`,
which is a numpy reimplementation of MiniLM and needs neither. The image carries
`artifacts/encoder/` (41 MB) instead of a 124 MB runtime. See README §9 and §14.

The encoder loads lazily on the first `/search` request, not at boot, and adds
about 101 MB resident once it does — worth knowing on a 512 MB free instance.
A deployment that omits `artifacts/encoder/` still serves every other endpoint;
`/search` returns `503` and `/health` reports `has_semantic_search: false`.

---

## CORS

The API sets `allow_origins=["*"]` with `allow_credentials=False`. The two
services are deployed to different hosts and are therefore always
cross-origin, and there are no accounts, no cookies, and no authenticated
state — so there is nothing an origin allowlist would protect. If
authentication is ever added, tighten this in `api/main.py` at the same time,
because `allow_credentials=True` with a wildcard origin is rejected by browsers
and is a genuine vulnerability if worked around.

---

## Verifying a deployment

```bash
API=https://<your-api>.onrender.com

curl -s $API/health

# Cold-start recommendations from three seeds (Toy Story, Pulp Fiction, Matrix)
curl -s -X POST $API/recommend \
  -H 'Content-Type: application/json' \
  -d '{"seed_movie_ids": [1, 296, 2571], "k": 5}'
```

Then open the frontend, pick three films, and confirm a slate comes back. If
the page shows **"Backend unreachable"**, it is telling you the exact URL it
tried — compare that against the API's real origin, and remember that fixing it
means rebuilding the frontend, not restarting it.
