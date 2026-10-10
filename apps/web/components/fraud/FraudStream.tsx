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
  enrichment_note: string;
  top_flagged: Flagged[];
}

const ACCENT = "#f43f5e";

function riskColor(r: number) {
  if (r >= 0.9) return "#dc2626";
  if (r >= 0.7) return "#ea580c";
  if (r >= 0.4) return "#d97706";
  return "#16a34a";
}

export function FraudStream() {
  const [data, setData] = useState<StreamSummary | null>(null);
  const [busy, setBusy] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(() => {
    setBusy(true);
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

  if (error)
    return (
      <div className="rounded-2xl border border-[#fecaca] bg-[#fef2f2] p-6 text-[13px] text-[#991b1b]">
        Stream unavailable: {error}
      </div>
    );

  if (!data)
    return (
      <div className="animate-pulse rounded-2xl border border-[#e7e0d5] bg-white p-6 text-[13px] text-[#a8a29e]">
        Scoring the demo stream…
      </div>
    );

  const caught = data.true_positives;
  const missed = data.missed_fraud;
  const precision = data.n_flagged ? caught / data.n_flagged : 0;

  return (
    <div className="grid gap-4 lg:grid-cols-[1fr_320px]">
      <div className="rounded-2xl border border-[#e7e0d5] bg-white p-6 shadow-sm">
        <div className="flex items-center justify-between">
          <h3 className="text-[15px] font-semibold tracking-tight">Highest-risk transactions</h3>
          <button
            onClick={load}
            disabled={busy}
            className="rounded-lg border border-[#e7e0d5] px-3 py-1.5 text-[12px] font-medium transition hover:bg-[#fafaf9] disabled:opacity-50"
          >
            {busy ? "Scoring…" : "Re-score stream"}
          </button>
        </div>

        <div className="mt-4 space-y-1.5">
          {data.top_flagged.map((t, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, x: -8 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: i * 0.04 }}
              className="flex items-center gap-3 rounded-xl border border-[#f5f5f4] px-3.5 py-2.5"
            >
              <span
                className="h-2 w-2 shrink-0 rounded-full"
                style={{ background: riskColor(t.risk) }}
              />
              <span className="w-24 shrink-0 font-mono text-[12px] tabular-nums">
                ${t.amount.toFixed(2)}
              </span>
              <div className="relative h-1.5 flex-1 overflow-hidden rounded-full bg-[#f5f5f4]">
                <motion.div
                  className="h-full rounded-full"
                  style={{ background: riskColor(t.risk) }}
                  initial={{ width: 0 }}
                  animate={{ width: `${Math.max(2, t.risk * 100)}%` }}
                  transition={{ delay: i * 0.04, duration: 0.5 }}
                />
              </div>
              <span className="w-14 shrink-0 text-right font-mono text-[11.5px] tabular-nums text-[#57534e]">
                {(t.risk * 100).toFixed(1)}%
              </span>
              {/* The label is the point: a monitor that hides its misses cannot be trusted. */}
              <span
                className="w-24 shrink-0 rounded-md px-2 py-0.5 text-center font-mono text-[10px] uppercase tracking-wider"
                style={
                  t.actual_fraud
                    ? { background: t.flagged ? "#dcfce7" : "#fef3c7", color: t.flagged ? "#166534" : "#92400e" }
                    : { background: "#f5f5f4", color: "#78716c" }
                }
              >
                {t.actual_fraud ? (t.flagged ? "caught" : "missed") : t.flagged ? "false alarm" : "legit"}
              </span>
              <span className="w-20 shrink-0 font-mono text-[10px] text-[#a8a29e]">{t.decisive}</span>
            </motion.div>
          ))}
        </div>

        <p className="mt-4 border-t border-[#f5f5f4] pt-3 text-[11.5px] leading-relaxed text-[#a8a29e]">
          {data.enrichment_note}
        </p>
      </div>

      <div className="space-y-3">
        <div className="rounded-2xl border border-[#e7e0d5] bg-white p-5 shadow-sm">
          <div className="font-mono text-[10px] uppercase tracking-[0.16em] text-[#a8a29e]">
            This batch
          </div>
          <div className="mt-2 text-3xl font-semibold tabular-nums tracking-tight">
            {data.n_scored.toLocaleString()}
          </div>
          <div className="text-[12px] text-[#57534e]">transactions scored</div>

          <dl className="mt-4 space-y-2 border-t border-[#f5f5f4] pt-3 text-[12.5px]">
            <Row label="Flagged" value={data.n_flagged} />
            <Row label="True positives" value={caught} color="#16a34a" />
            <Row label="False positives" value={data.false_positives} color="#d97706" />
            <Row label="Missed fraud" value={missed} color="#dc2626" />
          </dl>
        </div>

        <div className="rounded-2xl border border-[#e7e0d5] bg-white p-5 shadow-sm">
          <div className="font-mono text-[10px] uppercase tracking-[0.16em] text-[#a8a29e]">
            Operating point
          </div>
          <div className="mt-2 text-2xl font-semibold tabular-nums tracking-tight" style={{ color: ACCENT }}>
            {(data.threshold * 100).toFixed(0)}%
          </div>
          <div className="text-[12px] text-[#57534e]">risk threshold</div>
          <div className="mt-3 border-t border-[#f5f5f4] pt-3 text-[12px] text-[#57534e]">
            Batch precision{" "}
            <span className="font-mono tabular-nums text-[#1c1917]">{(precision * 100).toFixed(1)}%</span>
          </div>
        </div>
      </div>
    </div>
  );
}

function Row({ label, value, color }: { label: string; value: number; color?: string }) {
  return (
    <div className="flex items-baseline justify-between">
      <dt className="text-[#57534e]">{label}</dt>
      <dd className="font-mono tabular-nums font-medium" style={{ color: color ?? "#1c1917" }}>
        {value}
      </dd>
    </div>
  );
}
