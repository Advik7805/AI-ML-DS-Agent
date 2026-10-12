"use client";

import { useCallback, useEffect, useState } from "react";

import { clientFetch } from "@/lib/api";

/** Mirrors `drift_report()` in core/drift.py. */
interface DriftFeature {
  feature: string;
  psi: number;
  verdict: string;
}

interface DriftReport {
  n_features_scored: number;
  mean_psi: number;
  overall: string;
  n_drifted: number;
  n_watching: number;
  top: DriftFeature[];
  features_missing_from_batch?: string[];
  features_not_in_training?: string[];
  thresholds: { moderate: number; significant: number };
  caveats: string[];
  note?: string;
}

/**
 * Distribution-drift panel, shared by the fraud and maintenance bodies.
 *
 * Both are trained once and served indefinitely on an ephemeral container, which is the setup
 * where a model stays confident, keeps answering, and is quietly wrong. This makes the input
 * distribution visible instead of leaving it to be discovered during an incident.
 *
 * The verdict is rendered in the panel rather than inferred from the number here, so the
 * thresholds stay defined in exactly one place (core/drift.py).
 */
export function DriftPanel({
  tool,
  accent,
  label,
}: {
  /** Tool name, e.g. "check_fraud_drift". */
  tool: string;
  accent: string;
  label: string;
}) {
  const [data, setData] = useState<DriftReport | null>(null);
  const [busy, setBusy] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(() => {
    setBusy(true);
    setError(null);
    clientFetch<{ result: DriftReport }>(`/api/v1/agent/tools/${tool}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ arguments: {} }),
    })
      .then((d) => setData(d.result))
      .catch((e) => setError(e instanceof Error ? e.message : String(e)))
      .finally(() => setBusy(false));
  }, [tool]);

  useEffect(load, [load]);

  if (error) {
    return (
      <div className="rounded-2xl border border-line bg-void-3 p-5 text-[12.5px] text-ink-dim">
        Drift check unavailable ({error}).
      </div>
    );
  }

  if (busy || !data) {
    return (
      <div className="animate-pulse rounded-2xl border border-line bg-void-2 p-5 text-[12.5px] text-ink-faint">
        Comparing against the training distribution…
      </div>
    );
  }

  // Colour from the verdict thresholds, not from a hardcoded guess about what "bad" looks like.
  const sig = data.thresholds?.significant ?? 0.25;
  const mod = data.thresholds?.moderate ?? 0.1;
  const tone = {
    bar: accent,
    bg: "var(--color-void-3)",
    border: "var(--color-line)",
    text: "var(--color-ink)",
  };

  const pct = Math.min(100, (data.mean_psi / sig) * 100);

  return (
    <div className="rounded-2xl border border-line bg-void-2 p-5 shadow-sm">
      <div className="flex items-baseline justify-between gap-3">
        <div className="font-mono text-[10px] uppercase tracking-[0.16em] text-ink-faint">
          {label}
        </div>
        <button
          onClick={load}
          className="rounded-lg border border-line px-2.5 py-1 text-[11px] font-medium text-ink-dim transition hover:bg-white/5"
        >
          Recheck
        </button>
      </div>

      <div
        className="mt-3 rounded-xl border p-3"
        style={{ background: tone.bg, borderColor: tone.border }}
      >
        <div className="text-[13px] font-semibold" style={{ color: tone.text }}>
          {data.overall}
        </div>
        <div className="mt-1.5 h-1.5 overflow-hidden rounded-full bg-void-3">
          <div
            className="h-full rounded-full transition-all duration-700"
            style={{ width: `${pct}%`, background: tone.bar }}
          />
        </div>
        <div className="mt-1.5 font-mono text-[10.5px]" style={{ color: tone.text }}>
          mean PSI {data.mean_psi.toFixed(4)} · significant at {sig} · {data.n_drifted} drifted ·{" "}
          {data.n_watching} watching
        </div>
      </div>

      {data.top?.length > 0 && (
        <ul className="mt-3 space-y-1">
          {data.top.slice(0, 5).map((f) => (
            <li
              key={f.feature}
              className="flex items-baseline justify-between gap-2 font-mono text-[11px]"
            >
              <span className="truncate text-ink-dim">{f.feature}</span>
              <span
                className="tabular-nums"
                style={{
                  color: f.psi >= mod ? accent : "var(--color-ink-faint)",
                }}
              >
                {f.psi.toFixed(4)}
              </span>
            </li>
          ))}
        </ul>
      )}

      {data.features_missing_from_batch && data.features_missing_from_batch.length > 0 && (
        <p className="mt-3 rounded-lg bg-void-3 p-2 text-[10.5px] leading-relaxed text-ink-dim">
          Missing from this batch: {data.features_missing_from_batch.slice(0, 5).join(", ")}.
          That is a schema problem, not drift, and it is reported separately for that reason.
        </p>
      )}

      <p className="mt-3 border-t border-line pt-2.5 text-[10.5px] leading-relaxed text-ink-faint">
        Population Stability Index per feature, with bin edges cut on the training distribution
        and never moved — re-binning on incoming data would measure it against itself and never
        show drift. Computed on marginals, so a joint shift that leaves every marginal unchanged
        is invisible here. Drift means the model is outside the data it was trained on, not that
        it is necessarily wrong.
      </p>
    </div>
  );
}
