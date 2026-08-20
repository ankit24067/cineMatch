/**
 * The "Why this?" panel.
 *
 * These numbers are the actual per-component contributions captured during
 * scoring — not a plausible story reconstructed afterwards — so the panel
 * cannot disagree with the ranking it sits next to. On the cold-start path
 * `als` is exactly 0.0, which is worth showing rather than hiding: it is the
 * visible consequence of having no interaction history.
 */

import type { Explanation } from "../types";

const LABELS: Record<string, string> = {
  als: "Collaborative (latent)",
  item_cf: "Similar-viewer overlap",
  content: "Content similarity",
  popularity: "Popularity",
};

const BARS: Record<string, string> = {
  als: "bg-sky-400",
  item_cf: "bg-violet-400",
  content: "bg-emerald-400",
  popularity: "bg-amber-400",
};

const ORDER = ["als", "item_cf", "content", "popularity"];

export function WhyPanel({ explanation }: { explanation: Explanation }) {
  const entries = ORDER.filter((k) => k in explanation.contributions).map(
    (k) => [k, explanation.contributions[k]] as const,
  );

  return (
    <div className="space-y-3 rounded-lg border border-[var(--color-edge)] bg-[#0d0f14] p-4">
      <div>
        <h4 className="text-xs font-semibold tracking-wide text-neutral-300 uppercase">
          Why this film?
        </h4>
        <p className="mt-1 text-[11px] text-neutral-500">
          Share of the final score, measured while ranking.
        </p>
      </div>

      <div className="space-y-2">
        {entries.map(([key, value]) => (
          <div key={key}>
            <div className="flex items-baseline justify-between gap-2 text-[11px]">
              <span className="text-neutral-300">{LABELS[key] ?? key}</span>
              <span className="tabular-nums text-neutral-400">
                {(value * 100).toFixed(1)}%
              </span>
            </div>
            <div className="mt-1 h-1.5 overflow-hidden rounded-full bg-[#1c1f28]">
              <div
                className={`h-full rounded-full ${BARS[key] ?? "bg-neutral-400"}`}
                style={{ width: `${Math.max(value * 100, 0)}%` }}
              />
            </div>
          </div>
        ))}
      </div>

      {explanation.because_of && (
        <p className="border-t border-[var(--color-edge)] pt-3 text-xs text-neutral-400">
          Because you watched{" "}
          <span className="font-medium text-[var(--color-accent)]">
            {explanation.because_of.title}
          </span>
          <span className="text-neutral-600">
            {" "}
            · similarity {explanation.seed_similarity.toFixed(3)}
          </span>
        </p>
      )}

      {explanation.contributions.als === 0 && (
        <p className="text-[11px] text-neutral-500 italic">
          No collaborative signal — this is a cold-start recommendation built
          from your picks alone.
        </p>
      )}
    </div>
  );
}
