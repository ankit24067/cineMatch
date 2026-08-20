import { useEffect, useState } from "react";

import { api, ApiError } from "../api";
import type { SimilarResponse } from "../types";
import { Poster } from "./Poster";

interface Props {
  movieId: number;
  onClose: () => void;
  onUseAsSeed: (movieId: number) => void;
}

export function SimilarDrawer({ movieId, onClose, onUseAsSeed }: Props) {
  const [data, setData] = useState<SimilarResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setData(null);
    setError(null);

    api
      .similar(movieId, 10)
      .then((res) => !cancelled && setData(res))
      .catch(
        (err) =>
          !cancelled &&
          setError(err instanceof ApiError ? err.message : "Request failed."),
      );

    return () => {
      cancelled = true;
    };
  }, [movieId]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <div
      className="fixed inset-0 z-50 flex justify-end bg-black/70"
      onClick={onClose}
      role="presentation"
    >
      <aside
        className="h-full w-full max-w-md overflow-y-auto border-l border-[var(--color-edge)] bg-[var(--color-panel)] p-5"
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-label="Similar films"
      >
        <div className="mb-4 flex items-start justify-between gap-3">
          <h2 className="text-sm font-semibold">
            {data ? `More like ${data.movie.title}` : "More like this"}
          </h2>
          <button
            type="button"
            onClick={onClose}
            className="rounded border border-[var(--color-edge)] px-2 py-0.5 text-xs text-neutral-400 hover:text-white"
          >
            Close
          </button>
        </div>

        {error && <p className="text-xs text-red-300">{error}</p>}
        {!data && !error && (
          <p className="text-xs text-neutral-500">Loading…</p>
        )}

        {data && (
          <>
            {data.movie.overview && (
              <p className="mb-5 text-xs leading-relaxed text-neutral-400">
                {data.movie.overview}
              </p>
            )}
            <ul className="space-y-2">
              {data.similar.map((movie, i) => (
                <li
                  key={movie.movie_id}
                  className="flex items-center gap-3 rounded-lg border border-[var(--color-edge)] p-2"
                >
                  <Poster
                    title={movie.title}
                    poster={movie.poster}
                    className="h-16 w-11 shrink-0 rounded"
                  />
                  <div className="min-w-0 flex-1">
                    <p className="truncate text-xs font-medium">{movie.title}</p>
                    <p className="text-[11px] text-neutral-500">
                      {movie.year ?? "—"} · cosine{" "}
                      <span className="tabular-nums">
                        {data.similarities[i].toFixed(3)}
                      </span>
                    </p>
                  </div>
                  <button
                    type="button"
                    onClick={() => onUseAsSeed(movie.movie_id)}
                    className="shrink-0 rounded border border-[var(--color-edge)] px-2 py-1 text-[11px] text-neutral-300 hover:border-neutral-500 hover:text-white"
                  >
                    + Seed
                  </button>
                </li>
              ))}
            </ul>
          </>
        )}
      </aside>
    </div>
  );
}
