import { useState } from "react";

import type { Recommendation } from "../types";
import { Poster } from "./Poster";
import { WhyPanel } from "./WhyPanel";

interface Props {
  recommendation: Recommendation;
  onExplore: (movieId: number) => void;
}

export function RecommendationCard({ recommendation, onExplore }: Props) {
  const [showWhy, setShowWhy] = useState(false);
  const { movie, rank, score, explanation } = recommendation;

  return (
    <article className="flex w-[190px] shrink-0 flex-col gap-2">
      <div className="group relative overflow-hidden rounded-lg border border-[var(--color-edge)]">
        <Poster
          title={movie.title}
          poster={movie.poster}
          className="h-[280px] w-full"
        />
        <span className="absolute top-2 left-2 rounded bg-black/75 px-1.5 py-0.5 text-[11px] font-semibold tabular-nums">
          #{rank}
        </span>
        <button
          type="button"
          onClick={() => onExplore(movie.movie_id)}
          className="absolute inset-0 flex items-end justify-center bg-gradient-to-t from-black/85 to-transparent p-3 text-xs font-medium text-white opacity-0 transition group-hover:opacity-100 focus:opacity-100"
        >
          More like this
        </button>
      </div>

      <div className="min-h-[46px]">
        <h3 className="text-sm leading-tight font-medium">{movie.title}</h3>
        <p className="text-[11px] text-neutral-500">
          {movie.year ?? "—"} · {movie.genres.slice(0, 2).join(", ")}
        </p>
      </div>

      <button
        type="button"
        onClick={() => setShowWhy((v) => !v)}
        aria-expanded={showWhy}
        className="rounded border border-[var(--color-edge)] px-2 py-1 text-[11px] text-neutral-300 transition hover:border-neutral-500 hover:text-white"
      >
        {showWhy ? "Hide breakdown" : "Why this?"}
        <span className="ml-1 tabular-nums text-neutral-600">
          {score.toFixed(3)}
        </span>
      </button>

      {showWhy && <WhyPanel explanation={explanation} />}
    </article>
  );
}
