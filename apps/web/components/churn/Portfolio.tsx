"use client";

/**
 * Portfolio ranking — driven through the *agent* endpoint, not the module's own route.
 *
 * That is deliberate: this proves the hub's tool registry works from the browser. The request
 * goes to `/api/v1/agent/tools/rank_at_risk_customers`, which routes to whichever body owns the
 * tool. Swap the tool name and the same component drives a different body.
 */

import { motion } from "motion/react";
import { useEffect, useState } from "react";

import { rankAtRisk, type RankedCustomer, type RankedResponse } from "@/lib/api";

function riskColor(score: number) {
  if (score >= 0.75) return "#dc2626";
  if (score >= 0.5) return "#ea580c";
  if (score >= 0.25) return "#d97706";
  return "#16a34a";
}

export function Portfolio() {
  const [data, setData] = useState<RankedResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(true);
  const [n, setN] = useState(12);
  const [minScore, setMinScore] = useState(0.3);

  useEffect(() => {
    let cancelled = false;
    setBusy(true);
    rankAtRisk(n, minScore)
      .then((r) => {
        if (!cancelled) setData(r);
      })
      .catch((e) => {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      })
      .finally(() => {
        if (!cancelled) setBusy(false);
      });
    return () => {
      cancelled = true;
    };
  }, [n, minScore]);

  return (
    <div className="rounded-2xl border border-line bg-void-3 p-6 shadow-sm">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h3 className="text-[15px] font-semibold tracking-tight">Retention target list</h3>
          <p className="mt-1 text-[12.5px] text-ink-dim">
            {data
              ? `${data.n_returned} of ${data.n_scored.toLocaleString()} customers scored from ${data.data_source} data`
              : "Scoring the loaded customer base…"}
          </p>
          {data?.dataset && (
            <p className="mt-1 text-[11.5px] leading-relaxed text-ink-faint">
              {data.business_unit} · {data.dataset}
              {data.dataset_source ? ` — ${data.dataset_source}` : ""} · keyed by {data.join_key}.
              The same ids appear in the fraud board, so a subscriber here can be joined to their
              card activity.
            </p>
          )}
        </div>

        <div className="flex items-center gap-4">
          <label className="flex items-center gap-2 text-[12px] text-ink-dim">
            Top
            <select
              value={n}
              onChange={(e) => setN(Number(e.target.value))}
              className="rounded-md border border-line bg-void-3 px-2 py-1 text-[12px]"
            >
              {[5, 12, 25, 50].map((v) => (
                <option key={v} value={v}>
                  {v}
                </option>
              ))}
            </select>
          </label>
          <label className="flex items-center gap-2 text-[12px] text-ink-dim">
            Min risk
            <input
              type="range"
              min={0}
              max={0.95}
              step={0.05}
              value={minScore}
              onChange={(e) => setMinScore(Number(e.target.value))}
              className="h-1.5 w-24 cursor-pointer appearance-none rounded-full bg-line accent-[#f59e0b]"
            />
            <span className="w-9 font-mono tabular-nums">{minScore.toFixed(2)}</span>
          </label>
        </div>
      </div>

      {error && (
        <div className="mt-4 rounded-lg border border-red-500/30 bg-red-500/10 p-3 text-[12px] text-red-300">
          {error}
        </div>
      )}

      <div className="mt-5 overflow-x-auto">
        <table className="w-full text-left text-[12.5px]">
          <thead>
            <tr className="border-b border-line font-mono text-[10px] uppercase tracking-[0.14em] text-ink-faint">
              <th className="py-2 pr-3 font-normal">Customer</th>
              <th className="py-2 pr-3 font-normal">State</th>
              <th className="py-2 pr-3 font-normal">Risk</th>
              <th className="py-2 pr-3 font-normal">Band</th>
              <th className="py-2 pr-3 font-normal">Top driver</th>
              <th className="py-2 font-normal">Action</th>
            </tr>
          </thead>
          <tbody>
            {busy && !data && (
              <tr>
                <td colSpan={6} className="py-8 text-center text-[12px] text-ink-faint">
                  Loading…
                </td>
              </tr>
            )}
            {!busy && data?.customers.length === 0 && (
              <tr>
                <td colSpan={6} className="py-8 text-center text-[12px] text-ink-faint">
                  No customer clears that risk threshold.
                </td>
              </tr>
            )}
            {(data?.customers ?? []).map((c: RankedCustomer, i) => (
              <motion.tr
                key={`${c.row_index}-${i}`}
                initial={{ opacity: 0, y: 4 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: Math.min(i * 0.02, 0.4) }}
                className="border-b border-white/5 last:border-0 hover:bg-white/5"
              >
                <td className="py-2.5 pr-3 font-mono text-[11px]">
                  <span className="text-ink">{c.customer_id || `row ${c.row_index}`}</span>
                  <span className="ml-2 text-ink-faint">row {c.row_index}</span>
                </td>
                <td className="py-2.5 pr-3 font-mono text-[12px]">{c.state}</td>
                <td className="py-2.5 pr-3">
                  <div className="flex items-center gap-2">
                    <div className="h-1.5 w-20 overflow-hidden rounded-full bg-white/5">
                      <div
                        className="h-full rounded-full"
                        style={{ width: `${c.score * 100}%`, background: riskColor(c.score) }}
                      />
                    </div>
                    <span className="font-mono text-[11.5px] tabular-nums" style={{ color: riskColor(c.score) }}>
                      {(c.score * 100).toFixed(1)}%
                    </span>
                  </div>
                </td>
                <td className="py-2.5 pr-3 text-[12px] text-ink-dim">{c.label}</td>
                <td className="py-2.5 pr-3 font-mono text-[11px] text-ink-dim">{c.top_driver ?? "—"}</td>
                <td className="py-2.5 text-[11.5px] leading-snug text-ink-dim">{c.decision}</td>
              </motion.tr>
            ))}
          </tbody>
        </table>
      </div>

      {data && (
        <p className="mt-4 font-mono text-[10.5px] text-ink-faint">
          via POST /api/v1/agent/tools/rank_at_risk_customers · threshold {(data.threshold * 100).toFixed(0)}%
        </p>
      )}
    </div>
  );
}
