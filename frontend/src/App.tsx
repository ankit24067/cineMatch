import { useCallback, useEffect, useRef, useState } from "react";

import { api, ApiError, API_BASE } from "./api";
import { EvaluationPanel } from "./components/EvaluationPanel";
import { RecommendationCard } from "./components/RecommendationCard";
import { SeedPicker } from "./components/SeedPicker";
import { SemanticSearch } from "./components/SemanticSearch";
import { SimilarDrawer } from "./components/SimilarDrawer";
import type { Health, Movie, RecommendResponse } from "./types";

const DEFAULT_LAMBDA = 0.6;

export default function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);

  const [seeds, setSeeds] = useState<Movie[]>([]);
  const [result, setResult] = useState<RecommendResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [lambda, setLambda] = useState(DEFAULT_LAMBDA);

  const seedsRef = useRef(seeds);
  seedsRef.current = seeds;

  const [exploring, setExploring] = useState<number | null>(null);
  const [showResults, setShowResults] = useState(false);

  useEffect(() => {
    api
      .health()
      .then((h) => {
        setHealth(h);
        setLambda(h.mmr_lambda);
      })
      .catch((err) =>
        setHealthError(
          err instanceof ApiError ? err.message : "Cannot reach the API.",
        ),
      );
  }, []);

  const run = useCallback(
    async (movies: Movie[], mmrLambda: number) => {
      if (movies.length < 3) return;
      setBusy(true);
      setError(null);
      try {
        const response = await api.recommend({
          seed_movie_ids: movies.map((m) => m.movie_id),
          k: 20,
          mmr_lambda: mmrLambda,
        });
        setResult(response);
      } catch (err) {
        setError(
          err instanceof ApiError ? err.message : "Something went wrong.",
        );
      } finally {
        setBusy(false);
      }
    },
    [],
  );

  const hasResult = result !== null;
  useEffect(() => {
    if (!hasResult) return;
    const timer = setTimeout(() => void run(seedsRef.current, lambda), 300);
    return () => clearTimeout(timer);
  }, [lambda, hasResult, run]);

  async function addSeedById(movieId: number) {
    if (seeds.some((s) => s.movie_id === movieId)) return;
    try {
      const movie = await api.movie(movieId);
      setSeeds((prev) => [...prev, movie]);
      setExploring(null);
    } catch {}
  }

  return (
    <div className="min-h-full">
      <header className="border-b border-[var(--color-edge)] px-6 py-4">
        <div className="mx-auto flex max-w-6xl flex-wrap items-baseline justify-between gap-3">
          <div>
            <h1 className="text-lg font-semibold tracking-tight">
              Cine<span className="text-[var(--color-accent)]">Match</span>
            </h1>
            <p className="text-[11px] text-neutral-500">
              Hybrid, explainable film recommendations
            </p>
          </div>
          <div className="text-right text-[11px] text-neutral-500">
            {health ? (
              <>
                <span className="text-emerald-400">●</span>{" "}
                {health.n_items.toLocaleString()} films ·{" "}
                {health.has_interactions
                  ? `${health.n_users} users`
                  : "cold-start only"}
              </>
            ) : (
              <span className="text-neutral-600">connecting…</span>
            )}
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-6xl space-y-10 px-6 py-10">
        {healthError && (
          <div className="rounded-lg border border-red-900/60 bg-red-950/40 p-4 text-sm text-red-200">
            <p className="font-medium">Backend unreachable</p>
            <p className="mt-1 text-xs text-red-300/80">
              {healthError} Expected at{" "}
              <code className="text-red-200">{API_BASE}</code>. Start it with{" "}
              <code className="text-red-200">uvicorn api.main:app --reload</code>{" "}
              or point <code className="text-red-200">VITE_API_URL</code>{" "}
              elsewhere.
            </p>
          </div>
        )}

        <section className="space-y-2">
          <h2 className="text-2xl font-semibold tracking-tight">
            Tell us what you like.
          </h2>
          <p className="max-w-2xl text-sm leading-relaxed text-neutral-400">
            There is no account and no history here — every recommendation is
            built from the films you pick right now. Each card opens the real
            scoring breakdown behind it, including the parts that contributed
            nothing.
          </p>
        </section>

        <SeedPicker
          seeds={seeds}
          onChange={setSeeds}
          onSubmit={() => void run(seeds, lambda)}
          busy={busy}
        />

        {error && (
          <p className="rounded-lg border border-red-900/60 bg-red-950/40 px-4 py-3 text-sm text-red-200">
            {error}
          </p>
        )}

        {result && (
          <section className="space-y-4">
            <div className="flex flex-wrap items-end justify-between gap-4">
              <div>
                <h2 className="text-lg font-semibold">Recommended for you</h2>
                <p className="text-[11px] text-neutral-500">
                  {result.recommendations.length} films in{" "}
                  {result.took_ms.toFixed(1)} ms ·{" "}
                  {result.cold_start ? "cold start" : "warm user"}
                  {!result.personalised && (
                    <span className="text-amber-400">
                      {" "}
                      · too few picks to call this personalised
                    </span>
                  )}
                </p>
              </div>

              <label className="flex items-center gap-3 text-[11px] text-neutral-400">
                <span>
                  Diversity{" "}
                  <span className="tabular-nums text-neutral-300">
                    λ={lambda.toFixed(2)}
                  </span>
                </span>
                <input
                  type="range"
                  min={0.3}
                  max={1}
                  step={0.05}
                  value={lambda}
                  onChange={(e) => setLambda(Number(e.target.value))}
                  className="w-40 accent-[var(--color-accent)]"
                />
                <span className="w-24 text-neutral-600">
                  {lambda >= 0.99 ? "relevance only" : "more variety"}
                </span>
              </label>
            </div>

            <div className="rail flex gap-4 overflow-x-auto pb-4">
              {result.recommendations.map((rec) => (
                <RecommendationCard
                  key={rec.movie.movie_id}
                  recommendation={rec}
                  onExplore={setExploring}
                />
              ))}
            </div>
          </section>
        )}

        {/* Hidden rather than shown-and-failing when the deployment ships no
            encoder: /health tells us up front, so there is no reason to offer
            a box that can only return 503. */}
        {health?.has_semantic_search && (
          <SemanticSearch onExplore={setExploring} />
        )}

        <section className="space-y-3 border-t border-[var(--color-edge)] pt-8">
          <button
            type="button"
            onClick={() => setShowResults((v) => !v)}
            className="text-sm font-medium text-neutral-300 hover:text-white"
          >
            {showResults ? "− " : "+ "}
            How well does this actually work?
          </button>
          {showResults && <EvaluationPanel />}
        </section>
      </main>

      {exploring !== null && (
        <SimilarDrawer
          movieId={exploring}
          onClose={() => setExploring(null)}
          onUseAsSeed={(id) => void addSeedById(id)}
        />
      )}

      <footer className="border-t border-[var(--color-edge)] px-6 py-6 text-center text-[11px] text-neutral-600">
        Data: MovieLens ml-latest-small · Metadata &amp; posters: TMDB
        {health?.artifacts_built_at && (
          <> · Artifacts built {health.artifacts_built_at.slice(0, 10)}</>
        )}
      </footer>
    </div>
  );
}
