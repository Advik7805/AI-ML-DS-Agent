"use client";

/**
 * Fairness audit panel.
 *
 * The whole point of this body is the part most demos leave out: a model that screens for
 * diabetes and quietly misses one group. The panel leads with the verdict rather than the AUC,
 * and states plainly that the grouping is an age proxy — this dataset carries no race or gender
 * field, and inventing one would be the exact failure Fairlearn exists to prevent.
 */

import { motion } from "motion/react";
import { useEffect, useState } from "react";

import { clientFetch } from "@/lib/api";

interface Audit {
  grouping_variable: string;
  groups: Record<string, { false_negative_rate: number; false_positive_rate: number }>;
  fnr_gap: number;
  fpr_gap: number;
  worst_served_group: string;
  worst_served_fnr: number;
  verdict: string;
  metric_rationale: string;
}

const ACCENT = "#7c3aed";

export function FairnessAudit() {
  const [audit, setAudit] = useState<Audit | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showFpr, setShowFpr] = useState(false);

  useEffect(() => {
    clientFetch<{ result: Audit }>("/api/v1/agent/tools/audit_model_fairness", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ arguments: {} }),
    })
      .then((r) => setAudit(r.result))
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
  }, []);

  if (error)
    return (
      <div className="rounded-2xl border border-red-500/30 bg-red-500/10 p-6 text-[13px] text-red-300">
        Audit unavailable: {error}
      </div>
    );

  if (!audit)
    return (
      <div className="animate-pulse rounded-2xl border border-line bg-void-3 p-6 text-[13px] text-ink-faint">
        Running the audit…
      </div>
    );

  const rows = Object.entries(audit.groups);
  const worst = rows.reduce((a, b) => (b[1].false_negative_rate > a[1].false_negative_rate ? b : a));
  const breaches = audit.fnr_gap > 0.15;

  return (
    <div className="grid gap-4 lg:grid-cols-[1fr_320px]">
      <div className="rounded-2xl border border-line bg-void-3 p-6 shadow-sm">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <h3 className="text-[15px] font-semibold tracking-tight">Error rate by group</h3>
          <div className="flex gap-1 rounded-lg bg-white/5 p-0.5">
            {(["fnr", "fpr"] as const).map((m) => (
              <button
                key={m}
                onClick={() => setShowFpr(m === "fpr")}
                className={`rounded-md px-2.5 py-1 font-mono text-[10.5px] uppercase tracking-wider transition ${
                  (m === "fpr") === showFpr ? "bg-void-3 shadow-sm" : "text-ink-faint"
                }`}
              >
                {m}
              </button>
            ))}
          </div>
        </div>

        <p className="mt-1 text-[11.5px] text-ink-faint">
          {showFpr
            ? "False positive rate — healthy people wrongly told they are at risk."
            : "False negative rate — at-risk people the model missed. In screening, this is the harm that matters."}
        </p>

        <div className="mt-5 space-y-3">
          {rows.map(([group, vals], i) => {
            const value = showFpr ? vals.false_positive_rate : vals.false_negative_rate;
            const isWorst = group === worst[0] && !showFpr;
            return (
              <div key={group} className="flex items-center gap-3">
                <span className="w-20 shrink-0 font-mono text-[11.5px] text-ink-dim">{group}</span>
                <div className="relative h-6 flex-1 overflow-hidden rounded-lg bg-white/5">
                  <motion.div
                    className="h-full rounded-lg"
                    style={{ background: isWorst ? "#dc2626" : ACCENT, opacity: isWorst ? 1 : 0.72 }}
                    initial={{ width: 0 }}
                    animate={{ width: `${Math.max(2, value * 100)}%` }}
                    transition={{ delay: i * 0.06, type: "spring", stiffness: 120, damping: 18 }}
                  />
                  <span className="absolute inset-y-0 left-2 flex items-center font-mono text-[11px] font-medium tabular-nums text-white mix-blend-difference">
                    {(value * 100).toFixed(1)}%
                  </span>
                </div>
                {isWorst && (
                  <span className="shrink-0 rounded-md bg-red-500/10 px-2 py-0.5 font-mono text-[10px] uppercase tracking-wider text-[#dc2626]">
                    worst served
                  </span>
                )}
              </div>
            );
          })}
        </div>

        <div className="mt-5 border-t border-white/5 pt-3">
          <div className="font-mono text-[10px] uppercase tracking-[0.16em] text-ink-faint">
            Grouping
          </div>
          <p className="mt-1 text-[12.5px] leading-relaxed text-ink-dim">{audit.grouping_variable}</p>
        </div>
      </div>

      <div className="space-y-3">
        <div
          className="rounded-2xl border p-5 shadow-sm"
          style={{
            borderColor: breaches ? "rgba(239,68,68,0.30)" : "#bbf7d0",
            background: breaches ? "rgba(239,68,68,0.10)" : "#f0fdf4",
          }}
        >
          <div className="font-mono text-[10px] uppercase tracking-[0.16em] text-ink-faint">
            Verdict
          </div>
          <div
            className="mt-2 text-[15px] font-semibold leading-snug tracking-tight"
            style={{ color: breaches ? "#fca5a5" : "#6ee7b7" }}
          >
            {breaches ? "Do not deploy for screening" : "Within tolerance"}
          </div>
          <p className="mt-2 text-[12px] leading-relaxed" style={{ color: breaches ? "#fca5a5" : "#4ade80" }}>
            {audit.verdict}
          </p>
        </div>

        <div className="rounded-2xl border border-line bg-void-3 p-5 shadow-sm">
          <div className="font-mono text-[10px] uppercase tracking-[0.16em] text-ink-faint">
            Disparity
          </div>
          <div className="mt-2 flex items-baseline gap-2">
            <span
              className="text-3xl font-semibold tabular-nums tracking-tight"
              style={{ color: breaches ? "#dc2626" : ACCENT }}
            >
              {(audit.fnr_gap * 100).toFixed(1)}
            </span>
            <span className="text-[12.5px] text-ink-dim">point FNR gap</span>
          </div>
          <div className="mt-1 text-[11.5px] text-ink-dim">
            Tolerance 15.0 · FPR gap {(audit.fpr_gap * 100).toFixed(1)}
          </div>

          <div className="mt-3 border-t border-white/5 pt-3">
            <div className="text-[12px] text-ink-dim">
              Worst served:{" "}
              <span className="font-mono text-ink">{audit.worst_served_group}</span> at{" "}
              <span className="font-mono tabular-nums">{(audit.worst_served_fnr * 100).toFixed(1)}%</span>{" "}
              missed
            </div>
          </div>
        </div>

        <div className="rounded-2xl border border-line bg-void-3 p-5 shadow-sm">
          <div className="font-mono text-[10px] uppercase tracking-[0.16em] text-ink-faint">
            Why false negatives
          </div>
          <p className="mt-2 text-[11.5px] leading-relaxed text-ink-dim">
            {audit.metric_rationale}
          </p>
        </div>
      </div>
    </div>
  );
}
