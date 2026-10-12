"use client";

/**
 * Fleet urgency board.
 *
 * RUL is the one metric where the *distribution* matters more than the mean: an error of 20
 * cycles on an engine with 120 left is noise, and on an engine with 8 left it is a failure. The
 * board is therefore ordered by urgency, not by predicted life, and the error breakdown is
 * shown per remaining-life band beside it.
 */

import { motion } from "motion/react";
import { useEffect, useState } from "react";

import { clientFetch } from "@/lib/api";
import type { MetricsReport } from "@/lib/types";

interface Engine {
  unit: number;
  cycle: number;
  predicted_rul: number;
  urgency: number;
}

interface Fleet {
  n_engines: number;
  data_source: string;
  engines: Engine[];
}

const ACCENT = "#22d3ee";

function bandColor(_rul: number) {
  return ACCENT;
}

export function FleetBoard() {
  const [fleet, setFleet] = useState<Fleet | null>(null);
  const [metrics, setMetrics] = useState<MetricsReport | null>(null);
  const [n, setN] = useState(12);
  const [busy, setBusy] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    clientFetch<MetricsReport>("/api/v1/modules/maintenance/metrics")
      .then(setMetrics)
      .catch(() => setMetrics(null));
  }, []);

  useEffect(() => {
    let cancelled = false;
    setBusy(true);
    clientFetch<{ result: Fleet }>("/api/v1/agent/tools/rank_engines_by_urgency", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ arguments: { n } }),
    })
      .then((r) => !cancelled && setFleet(r.result))
      .catch((e) => !cancelled && setError(e instanceof Error ? e.message : String(e)))
      .finally(() => !cancelled && setBusy(false));
    return () => {
      cancelled = true;
    };
  }, [n]);

  if (error)
    return (
      <div className="rounded-2xl border border-line bg-void-3 p-6 text-[13px] text-ink-dim">
        Fleet unavailable: {error}
      </div>
    );

  // Band MAEs live in the metric names, e.g. mae_0_30_cycles. Pulling them from the report
  // rather than hard-coding keeps the panel honest if the bands ever change.
  const bands = Object.entries(metrics?.scores ?? {})
    .filter(([k]) => /^mae_\d+_\d+_cycles$/.test(k))
    .map(([k, v]) => {
      const m = k.match(/^mae_(\d+)_(\d+)_cycles$/)!;
      return { label: `${m[1]}–${m[2]}`, v };
    });

  return (
    <div className="grid gap-4 lg:grid-cols-[1fr_300px]">
      <div className="rounded-2xl border border-line bg-void-2 p-6 shadow-sm">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h3 className="text-[15px] font-semibold tracking-tight">Maintenance queue</h3>
            <p className="mt-0.5 text-[11.5px] text-ink-faint">
              {fleet ? `${fleet.n_engines} engines · ${fleet.data_source} telemetry` : "loading…"}
            </p>
          </div>
          <div className="flex items-center gap-2">
            <span className="font-mono text-[10.5px] text-ink-faint">show</span>
            <input
              type="range"
              min={5}
              max={25}
              value={n}
              onChange={(e) => setN(Number(e.target.value))}
              className="w-28"
              style={{ accentColor: ACCENT }}
            />
            <span className="w-6 font-mono text-[11px] tabular-nums">{n}</span>
          </div>
        </div>

        <div className="mt-4 space-y-1">
          <div className="grid grid-cols-[52px_70px_1fr_86px_64px] gap-2 px-1 pb-1 font-mono text-[10px] uppercase tracking-wider text-ink-faint">
            <span>unit</span>
            <span>cycle</span>
            <span>predicted life left</span>
            <span className="text-right">rul</span>
            <span className="text-right">urgency</span>
          </div>

          {(fleet?.engines ?? []).map((e, i) => (
            <motion.div
              key={e.unit}
              initial={{ opacity: 0, y: 4 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.025 }}
              className="grid grid-cols-[52px_70px_1fr_86px_64px] items-center gap-2 rounded-lg px-1 py-1.5 hover:bg-white/5"
            >
              <span className="font-mono text-[12px] text-ink-dim">#{e.unit}</span>
              <span className="font-mono text-[11.5px] tabular-nums text-ink-dim">{e.cycle}</span>
              <div className="relative h-2 overflow-hidden rounded-full bg-void-3">
                <motion.div
                  className="h-full rounded-full"
                  style={{ background: bandColor(e.predicted_rul) }}
                  initial={{ width: 0 }}
                  animate={{ width: `${Math.min(100, Math.max(3, e.urgency * 100))}%` }}
                  transition={{ delay: i * 0.025, duration: 0.45 }}
                />
              </div>
              <span
                className="text-right font-mono text-[12px] font-medium tabular-nums"
                style={{ color: bandColor(e.predicted_rul) }}
              >
                {e.predicted_rul.toFixed(0)} cyc
              </span>
              <span className="text-right font-mono text-[11px] tabular-nums text-ink-faint">
                {e.urgency.toFixed(2)}
              </span>
            </motion.div>
          ))}

          {busy && (
            <div className="py-6 text-center text-[12px] text-ink-faint">Ranking the fleet…</div>
          )}
        </div>
      </div>

      <div className="space-y-3">
        {metrics && (
          <div className="rounded-2xl border border-line bg-void-2 p-5 shadow-sm">
            <div className="font-mono text-[10px] uppercase tracking-[0.16em] text-ink-faint">
              Error by remaining life
            </div>
            <p className="mt-2 text-[11.5px] leading-relaxed text-ink-dim">
              Overall MAE hides where the model fails. These are cycles of error inside each band,
              cross-validated by engine.
            </p>
            <div className="mt-3 space-y-2">
              {bands.map((b) => {
                const max = Math.max(...bands.map((x) => x.v), 1);
                return (
                  <div key={b.label} className="flex items-center gap-2">
                    <span className="w-14 shrink-0 font-mono text-[10.5px] text-ink-dim">
                      {b.label}
                    </span>
                    <div className="relative h-2 flex-1 overflow-hidden rounded-full bg-void-3">
                      <div
                        className="h-full rounded-full"
                        style={{ width: `${(b.v / max) * 100}%`, background: ACCENT }}
                      />
                    </div>
                    <span className="w-12 shrink-0 text-right font-mono text-[11px] tabular-nums">
                      {b.v.toFixed(1)}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {metrics && (
          <div className="rounded-2xl border border-line bg-void-2 p-5 shadow-sm">
            <div className="font-mono text-[10px] uppercase tracking-[0.16em] text-ink-faint">
              Model
            </div>
            <div className="mt-2 text-3xl font-semibold tabular-nums tracking-tight">
              {metrics.scores.r2?.toFixed(3)}
            </div>
            <div className="text-[12px] text-ink-dim">R² · MAE {metrics.scores.mae_cycles?.toFixed(1)} cycles</div>
            <p className="mt-3 border-t border-line pt-3 text-[11px] leading-relaxed text-ink-faint">
              {metrics.notes.find((x) => /source/i.test(x)) ?? "Synthetic CMAPSS-compatible telemetry."}
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
