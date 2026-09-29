import { useRef, useState } from "react";

import { api, ApiError } from "../api";
import type { SearchResponse } from "../types";
import { Poster } from "./Poster";

const EXAMPLES = [
  "space movies that are actually about grief",
  "dystopian future ruled by machines",
  "a heist that goes wrong",
];

interface Props {
  onExplore: (movieId: number) => void;
}

/**
 * Free-text search over the content embeddings.
 *
 * Deliberately submit-on-enter rather than debounced-as-you-type like the seed
 * picker. That search is a title substring match the server answers from a
 * pandas mask; this one runs six transformer blocks per call, and the first
 * call of a container's life also pays ~0.5 s to load the encoder. Firing one
 * per keystroke would be a very different load profile for a feature where the
 * user is composing a sentence, not a prefix.
 */
export function SemanticSearch({ onExplore }: Props) {
  const [query, setQuery] = useState("");
  const [response, setResponse] = useState<SearchResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const latest = useRef(0);

  async function submit(raw: string) {
    const trimmed = raw.trim();
    if (trimmed.length < 2) return;

    setBusy(true);
    setError(null);
    const ticket = ++latest.current;
    try {
      const body = await api.semanticSearch(trimmed, 12);
      if (ticket !== latest.current) return;
      setResponse(body);
    } catch (err) {
      if (ticket !== latest.current) return;
      setResponse(null);
      setError(
        err instanceof ApiError ? err.message : "Search failed.",
      );
    } finally {
      if (ticket === latest.current) setBusy(false);
    }
  }

  return (
    <section className="space-y-4 border-t border-[var(--color-edge)] pt-8">
      <div>
        <h2 className="text-lg font-semibold">Search by what a film is about</h2>
        <p className="mt-1 max-w-2xl text-[11px] leading-relaxed text-neutral-500">
          Describe a film rather than naming it. The query is encoded into the
          same space as the catalogue&rsquo;s plot summaries, so results need
          share no word with what you typed. It matches on story, not on
          craft &mdash; asking about cinematography or pacing has little to
          match against.
        </p>
      </div>

      <form
        onSubmit={(e) => {
          e.preventDefault();
          void submit(query);
        }}
        className="flex gap-2"
      >
        <input
          id="semantic-search"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Try “a lonely robot cleaning an abandoned planet”…"
          autoComplete="off"
          aria-label="Describe a film"
          className="min-w-0 flex-1 rounded-lg border border-[var(--color-edge)] bg-[var(--color-panel)] px-4 py-3 text-sm outline-none placeholder:text-neutral-600 focus:border-neutral-500"
        />
        <button
          type="submit"
          disabled={busy || query.trim().length < 2}
          className="shrink-0 rounded-lg border border-[var(--color-edge)] px-4 py-3 text-sm font-medium transition enabled:hover:border-neutral-500 disabled:cursor-not-allowed disabled:text-neutral-600"
        >
          {busy ? "Searching…" : "Search"}
        </button>
      </form>

      <ul className="flex flex-wrap gap-2">
        {EXAMPLES.map((example) => (
          <li key={example}>
            <button
              type="button"
              onClick={() => {
                setQuery(example);
                void submit(example);
              }}
              className="rounded-full border border-[var(--color-edge)] px-3 py-1 text-[11px] text-neutral-400 transition hover:border-neutral-500 hover:text-neutral-200"
            >
              {example}
            </button>
          </li>
        ))}
      </ul>

      {error && (
        <p className="rounded-lg border border-red-900/60 bg-red-950/40 px-3 py-2 text-xs text-red-300">
          {error}
        </p>
      )}

      {response && (
        <div className="space-y-3">
          <p className="text-[11px] text-neutral-500">
            {response.results.length} films in {response.took_ms.toFixed(1)} ms
            {" · "}
            <span className="text-neutral-600">
              cosine, not a probability &mdash; a query nothing sits near still
              returns its nearest neighbours
            </span>
          </p>

          <ul className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-4">
            {response.results.map((movie, i) => (
              <li key={movie.movie_id}>
                <button
                  type="button"
                  onClick={() => onExplore(movie.movie_id)}
                  className="flex w-full items-center gap-2 rounded-lg border border-[var(--color-edge)] p-2 text-left transition hover:border-neutral-500"
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
                      {movie.year ?? "—"} ·{" "}
                      <span className="tabular-nums">
                        {response.scores[i].toFixed(3)}
                      </span>
                    </span>
                  </span>
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}
    </section>
  );
}
