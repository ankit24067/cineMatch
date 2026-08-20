# CineMatch serving image.
#
# Serving does no training and no encoding — it is matrix slices and lookups
# over precomputed artifacts — so this image carries neither torch nor an ONNX
# runtime. See README section 14.
#
# The MovieLens licence forbids redistributing the raw ratings, so data/raw is
# absent here by design. api/artifacts.py detects that at boot, degrades to an
# empty interaction matrix, and serves the cold-start path. /health reports the
# mode as has_interactions: false, and /recommend with a user_id returns 409.

FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies first: this layer is cached until requirements.txt changes,
# so an artifact rebuild does not reinstall scipy and pandas.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Only what serving reads. .dockerignore drops artifacts/tmdb.json (4.1 MB,
# build-time metadata cache) and documents_sample.json — neither is loaded by
# the API; the catalogue already carries the posters and overviews.
COPY api/ ./api/
COPY artifacts/ ./artifacts/
# scripts/ is imported lazily by api/artifacts.py, and only when data/raw is
# mounted. Copied so a deployment that does mount the ratings still works.
COPY scripts/ ./scripts/

# Run unprivileged. Nothing here needs write access at runtime.
RUN useradd --create-home --uid 10001 cinematch && chown -R cinematch:cinematch /app
USER cinematch

EXPOSE 8000

# Render, Fly and Cloud Run all inject $PORT; default to 8000 for plain
# `docker run`. Single worker: the artifacts are ~21 MB of numpy held per
# process, so a second worker doubles memory for no throughput gain on a
# free-tier instance.
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
