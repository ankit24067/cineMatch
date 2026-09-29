import { useEffect, useRef, useState } from "react";

import { api, ApiError } from "../api";
import type { Movie } from "../types";
import { Poster } from "./Poster";

const MIN_SEEDS = 3;
const MAX_SEEDS = 8;

interface Props {
  seeds: Movie[];
  onChange: (seeds: Movie[]) => void;
  onSubmit: () => void;
  busy: boolean;
}

export function SeedPicker({ seeds, onChange, onSubmit, busy }: Props) {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<Movie[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [searching, setSearching] = useState(false);
  const latest = useRef(0);

  useEffect(() => {
    const trimmed = query.trim();
    if (trimmed.length < 2) {
      setResults([]);
      setSearching(false);
      return;
    }

    setSearching(true);
    const ticket = ++latest.current;
    const timer = setTimeout(async () => {
      try {
        const hits = await api.titleSearch(trimmed, 12);

        if (ticket !== latest.current) return;
        setResults(hits);
        setError(null);
      } catch (err) {
        if (ticket !== latest.current) return;
        setError(err instanceof ApiError ? err.message : "Search failed.");
        setResults([]);
      } finally {
        if (ticket === latest.current) setSearching(false);
      }
    }, 250);

    return () => clearTimeout(timer);
  }, [query]);

  const chosen = new Set(seeds.map((s) => s.movie_id));

  function toggle(movie: Movie) {
    if (chosen.has(movie.movie_id)) {
      onChange(seeds.filter((s) => s.movie_id !== movie.movie_id));
    } else if (seeds.length < MAX_SEEDS) {
      onChange([...seeds, movie]);
    }
  }

  const enough = seeds.length >= MIN_SEEDS;

  return (
    <section className="mx-auto w-full max-w-3xl space-y-5">
      <div>
        <label
          htmlFor="seed-search"
          className="mb-2 block text-sm text-neutral-300"
        >
          Search the catalogue and pick at least {MIN_SEEDS} films you like.
        </label>
        <input
          id="seed-search"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Try “matrix”, “godfather”, “spirited away”…"
          autoComplete="off"
          className="w-full rounded-lg border border-[var(--color-edge)] bg-[var(--color-panel)] px-4 py-3 text-sm outline-none placeholder:text-neutral-600 focus:border-neutral-500"
        />
      </div>

      {error && (
        <p className="rounded-lg border border-red-900/60 bg-red-950/40 px-3 py-2 text-xs text-red-300">
          {error}
        </p>
      )}

      {searching && !results.length && (
        <p className="text-xs text-neutral-500">Searching…</p>
      )}

      {results.length > 0 && (
        <ul className="grid grid-cols-2 gap-2 sm:grid-cols-3">
          {results.map((movie) => {
            const picked = chosen.has(movie.movie_id);
            const full = !picked && seeds.length >= MAX_SEEDS;
            return (
              <li key={movie.movie_id}>
                <button
                  type="button"
                  disabled={full}
                  onClick={() => toggle(movie)}
                  className={`flex w-full items-center gap-2 rounded-lg border p-2 text-left transition ${
                    picked
                      ? "border-[var(--color-accent)] bg-amber-500/10"
                      : "border-[var(--color-edge)] hover:border-neutral-500"
                  } ${full ? "cursor-not-allowed opacity-40" : ""}`}
                >
                  <Poster
                    title={movie.title}
                    poster={movie.poster}
                    className="h-14 w-10 shrink-0 rounded"
                  />
                  <span className="min-w-0">
                    <span className="block truncate text-xs font-medium">
                      {movie.title}
                    </span>
                    <span className="block text-[11px] text-neutral-500">
                      {movie.year ?? "—"}
                    </span>
                  </span>
                </button>
              </li>
            );
          })}
        </ul>
      )}

      <div className="rounded-lg border border-[var(--color-edge)] bg-[var(--color-panel)] p-4">
        <div className="mb-3 flex items-center justify-between">
          <h2 className="text-xs font-semibold tracking-wide text-neutral-300 uppercase">
            Your picks — {seeds.length}/{MAX_SEEDS}
          </h2>
          {seeds.length > 0 && (
            <button
              type="button"
              onClick={() => onChange([])}
              className="text-[11px] text-neutral-500 hover:text-neutral-300"
            >
              Clear
            </button>
          )}
        </div>

        {seeds.length === 0 ? (
          <p className="text-xs text-neutral-600">Nothing picked yet.</p>
        ) : (
          <ul className="flex flex-wrap gap-2">
            {seeds.map((movie) => (
              <li key={movie.movie_id}>
                <button
                  type="button"
                  onClick={() => toggle(movie)}
                  title="Remove"
                  className="flex items-center gap-1.5 rounded-full border border-[var(--color-edge)] py-1 pr-2 pl-3 text-xs hover:border-red-800 hover:text-red-300"
                >
                  {movie.title}
                  <span aria-hidden className="text-neutral-600">
                    ×
                  </span>
                </button>
              </li>
            ))}
          </ul>
        )}

        <button
          type="button"
          disabled={!enough || busy}
          onClick={onSubmit}
          className="mt-4 w-full rounded-lg bg-[var(--color-accent)] px-4 py-2.5 text-sm font-semibold text-black transition enabled:hover:brightness-110 disabled:cursor-not-allowed disabled:bg-neutral-800 disabled:text-neutral-500"
        >
          {busy
            ? "Ranking…"
            : enough
              ? "Get recommendations"
              : `Pick ${MIN_SEEDS - seeds.length} more`}
        </button>
      </div>
    </section>
  );
}
