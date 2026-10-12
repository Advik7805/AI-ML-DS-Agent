"use client";

/**
 * Live fraud monitor.
 *
 * Driven through `/api/v1/agent/tools/summarise_fraud_stream`, so this exercises the hub's tool
 * registry from the browser rather than the module's own route. The demo stream is *labelled*:
 * every transaction carries its true outcome, which is what lets the panel show the misses and
 * not just the catches. A monitor that only displays detections is marketing, not monitoring.
 */

import { motion } from "motion/react";
import { useCallback, useEffect, useState } from "react";

import { WakeRetry } from "@/components/body/WakeRetry";
import { clientFetch } from "@/lib/api";

interface Flagged {
  amount: number;
  risk: number;
  flagged: boolean;
  actual_fraud: boolean;
  decisive: string;
}

interface StreamSummary {
  n_scored: number;
  n_flagged: number;
  true_positives: number;
  false_positives: number;
  missed_fraud: number;
  threshold: number;
  enriched: boolean;
  data_source: string;
  enrichment_note: string;
  top_flagged: Flagged[];
}

const ACCENT = "#f43f5e";

function riskColor(_r: number) {
  return ACCENT;
}

export function FraudStream() {
  const [data, setData] = useState<StreamSummary | null>(null);
  const [busy, setBusy] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(() => {
    setBusy(true);
    setError(null);
    clientFetch<{ result: StreamSummary }>("/api/v1/agent/tools/summarise_fraud_stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ arguments: {} }),
    })
      .then((r) => setData(r.result))
      .catch((e) => setError(e instanceof Error ? e.message : String(e)))
      .finally(() => setBusy(false));
  }, []);

  useEffect(load, [load]);

  if (error) return <WakeRetry error={`Stream unavailable: ${error}`} onRetry={load} busy={busy} label="Retry stream" />;

  if (!data)
    return (
      <div className="animate-pulse rounded-2xl border border-line bg-void-2 p-6 text-[13px] text-ink-faint">
        Scoring the demo stream…
      </div>
    );

  const caught = data.true_positives;
  const missed = data.missed_fraud;
  const precision = data.n_flagged ? caught / data.n_flagged : 0;

  return (
    <div className="grid gap-4 lg:grid-cols-[1fr_320px]">
      <div className="rounded-2xl border border-line bg-void-2 p-6 shadow-sm">
        <div className="flex items-center justify-between">
          <h3 className="text-[15px] font-semibold tracking-tight">Highest-risk transactions</h3>
          <button
            onClick={load}
            disabled={busy}
            className="rounded-lg border border-line px-3 py-1.5 text-[12px] font-medium transition hover:bg-white/5 disabled:opacity-50"
          >
            {busy ? "Scoring…" : "Re-score stream"}
          </button>
        </div>

        {busy && <p role="status" className="mt-3 text-[11px] text-ink-faint">Scoring transactions through the fraud ensemble…</p>}
        <div className="mt-4 space-y-1.5">
          {data.top_flagged.map((t, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, x: -8 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: i * 0.04 }}
              className="flex items-center gap-3 rounded-xl border border-line px-3.5 py-2.5"
            >
              <span
                className="h-2 w-2 shrink-0 rounded-full"
                style={{ background: riskColor(t.risk) }}
              />
              <span className="w-24 shrink-0 font-mono text-[12px] tabular-nums">
                ${t.amount.toFixed(2)}
              </span>
              <div className="relative h-1.5 flex-1 overflow-hidden rounded-full bg-void-3">
                <motion.div
                  className="h-full rounded-full"
                  style={{ background: riskColor(t.risk) }}
                  initial={{ width: 0 }}
                  animate={{ width: `${Math.max(2, t.risk * 100)}%` }}
                  transition={{ delay: i * 0.04, duration: 0.5 }}
                />
              </div>
              <span className="w-14 shrink-0 text-right font-mono text-[11.5px] tabular-nums text-ink-dim">
                {(t.risk * 100).toFixed(1)}%
              </span>
              {/* The label is the point: a monitor that hides its misses cannot be trusted. */}
              <span
                className="w-24 shrink-0 rounded-md px-2 py-0.5 text-center font-mono text-[10px] uppercase tracking-wider"
                style={{
                  background: "var(--color-void-3)",
                  color: t.actual_fraud ? ACCENT : "var(--color-ink-dim)",
                }}
              >
                {t.actual_fraud ? (t.flagged ? "caught" : "missed") : t.flagged ? "false alarm" : "legit"}
              </span>
              <span className="w-20 shrink-0 font-mono text-[10px] text-ink-faint">{t.decisive}</span>
            </motion.div>
          ))}
        </div>

        <div className="mt-4 border-t border-line pt-3 text-[11.5px] leading-relaxed text-ink-faint">
          <p><span className="font-mono text-[9px] uppercase tracking-wider">Data source · </span>{data.data_source}</p>
          <p className="mt-1">{data.enrichment_note}</p>
        </div>
      </div>

      <div className="space-y-3">
        <div className="rounded-2xl border border-line bg-void-2 p-5 shadow-sm">
          <div className="font-mono text-[10px] uppercase tracking-[0.16em] text-ink-faint">
            This batch
          </div>
          <div className="mt-2 text-3xl font-semibold tabular-nums tracking-tight">
            {data.n_scored.toLocaleString()}
          </div>
          <div className="text-[12px] text-ink-dim">transactions scored</div>

          <dl className="mt-4 space-y-2 border-t border-line pt-3 text-[12.5px]">
            <Row label="Flagged" value={data.n_flagged} />
            <Row label="True positives" value={caught} color={ACCENT} />
            <Row label="False positives" value={data.false_positives} />
            <Row label="Missed fraud" value={missed} color={ACCENT} />
          </dl>
        </div>

        <div className="rounded-2xl border border-line bg-void-2 p-5 shadow-sm">
          <div className="font-mono text-[10px] uppercase tracking-[0.16em] text-ink-faint">
            Operating point
          </div>
          <div className="mt-2 text-2xl font-semibold tabular-nums tracking-tight" style={{ color: ACCENT }}>
            {(data.threshold * 100).toFixed(0)}%
          </div>
          <div className="text-[12px] text-ink-dim">risk threshold</div>
          <div className="mt-3 border-t border-line pt-3 text-[12px] text-ink-dim">
            Batch precision{" "}
            <span className="font-mono tabular-nums text-ink">{(precision * 100).toFixed(1)}%</span>
          </div>
        </div>
      </div>
    </div>
  );
}

function Row({ label, value, color }: { label: string; value: number; color?: string }) {
  return (
    <div className="flex items-baseline justify-between">
      <dt className="text-ink-dim">{label}</dt>
      <dd className="font-mono tabular-nums font-medium" style={{ color: color ?? "var(--color-ink)" }}>
        {value}
      </dd>
    </div>
  );
}
