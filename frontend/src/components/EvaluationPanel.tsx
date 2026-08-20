import { useEffect, useState } from "react";

import { api } from "../api";
import type { Evaluation } from "../types";

export function EvaluationPanel() {
  const [data, setData] = useState<Evaluation | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    api
      .evaluation()
      .then(setData)
      .catch(() => setFailed(true));
  }, []);

  if (failed) return null;
  if (!data) return <p className="text-xs text-neutral-600">Loading results…</p>;

  const best = Math.max(...data.results.map((r) => r.ndcg));

  return (
    <div className="space-y-3">
      <div className="overflow-x-auto rounded-lg border border-[var(--color-edge)]">
        <table className="w-full min-w-[520px] text-left text-xs">
          <thead className="bg-[#0d0f14] text-neutral-400">
            <tr>
              <th className="px-3 py-2 font-medium">Model</th>
              <th className="px-3 py-2 text-right font-medium">NDCG@10</th>
              <th className="px-3 py-2 text-right font-medium">MAP@10</th>
              <th className="px-3 py-2 text-right font-medium">Coverage</th>
              <th className="px-3 py-2 text-right font-medium">ILD</th>
              <th className="px-3 py-2 text-right font-medium">p95</th>
            </tr>
          </thead>
          <tbody>
            {data.results.map((row) => {
              const shipped = row.name.includes("shipped");
              return (
                <tr
                  key={row.name}
                  className={`border-t border-[var(--color-edge)] ${
                    shipped ? "bg-amber-500/5 text-white" : "text-neutral-300"
                  }`}
                >
                  <td className="px-3 py-2">
                    {row.name}
                    {row.ndcg === best && (
                      <span className="ml-1.5 text-[10px] text-[var(--color-accent)]">
                        best
                      </span>
                    )}
                  </td>
                  <td className="px-3 py-2 text-right tabular-nums">
                    {row.ndcg.toFixed(4)}
                  </td>
                  <td className="px-3 py-2 text-right tabular-nums">
                    {row.map_.toFixed(4)}
                  </td>
                  <td className="px-3 py-2 text-right tabular-nums">
                    {row.coverage.toFixed(1)}%
                  </td>
                  <td className="px-3 py-2 text-right tabular-nums">
                    {row.diversity.toFixed(3)}
                  </td>
                  <td className="px-3 py-2 text-right tabular-nums">
                    {row.p95_ms.toFixed(0)}ms
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <p className="text-[11px] leading-relaxed text-neutral-500">
        Held-out temporal split, k = 10, {data.results[0]?.n_users} test users.
        Weights{" "}
        {Object.entries(data.config.weights)
          .map(([k, v]) => `${k} ${v}`)
          .join(" · ")}
        , λ = {data.config.lambda}. Coverage is the share of the catalogue that
        reaches anyone — at {data.results.at(-1)?.coverage.toFixed(1)}% most of
        the library is never recommended, which is the clearest weakness in
        these numbers.
      </p>
    </div>
  );
}
