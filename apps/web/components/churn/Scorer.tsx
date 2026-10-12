"use client";

/**
 * The scorer — the interactive heart of the churn body.
 *
 * Sliders for the 13 model features, a live score, and SHAP drivers underneath. The explanation
 * panel labels its axis as log-odds because that is what TreeSHAP is exact in; implying the
 * bars sum to the displayed probability would be a lie the backend explicitly warns against.
 */

import { motion } from "motion/react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { WakeRetry } from "@/components/body/WakeRetry";
import { explain, predict } from "@/lib/api";
import type { Explanation, Prediction } from "@/lib/types";

interface FeatureSpec {
  key: string;
  label: string;
  min: number;
  max: number;
  step: number;
  value: number;
  group: "usage" | "plan" | "support";
}

const FEATURES: FeatureSpec[] = [
  { key: "account_length", label: "Account length (months)", min: 1, max: 243, step: 1, value: 120, group: "support" },
  { key: "total_day_minutes", label: "Day minutes", min: 0, max: 350, step: 0.1, value: 180, group: "usage" },
  { key: "total_day_calls", label: "Day calls", min: 0, max: 165, step: 1, value: 110, group: "usage" },
  { key: "total_eve_minutes", label: "Evening minutes", min: 0, max: 360, step: 0.1, value: 200, group: "usage" },
  { key: "total_eve_calls", label: "Evening calls", min: 0, max: 170, step: 1, value: 100, group: "usage" },
  { key: "total_night_minutes", label: "Night minutes", min: 0, max: 395, step: 0.1, value: 200, group: "usage" },
  { key: "total_night_calls", label: "Night calls", min: 0, max: 175, step: 1, value: 100, group: "usage" },
  { key: "total_intl_minutes", label: "International minutes", min: 0, max: 20, step: 0.1, value: 10, group: "usage" },
  { key: "total_intl_calls", label: "International calls", min: 0, max: 20, step: 1, value: 4, group: "usage" },
  { key: "number_customer_service_calls", label: "Service calls", min: 0, max: 9, step: 1, value: 1, group: "support" },
  { key: "number_vmail_messages", label: "Voicemail messages", min: 0, max: 51, step: 1, value: 0, group: "support" },
  { key: "international_plan", label: "International plan", min: 0, max: 1, step: 1, value: 0, group: "plan" },
  { key: "voice_mail_plan", label: "Voice-mail plan", min: 0, max: 1, step: 1, value: 0, group: "plan" },
];

const DEFAULTS = Object.fromEntries(FEATURES.map((f) => [f.key, f.value]));

/** A profile the data says is dangerous: many service calls + international plan. */
const ANGRY = { ...DEFAULTS, number_customer_service_calls: 6, international_plan: 1, total_day_minutes: 250 };

function riskColor(_score: number) {
  return "var(--color-churn)";
}

export function Scorer() {
  const [values, setValues] = useState<Record<string, number>>(DEFAULTS);
  const [pred, setPred] = useState<Prediction | null>(null);
  const [expl, setExpl] = useState<Explanation | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const run = useCallback(async (feats: Record<string, number>) => {
    setBusy(true);
    setError(null);
    try {
      // Both calls in parallel — they hit the same record and neither depends on the other.
      const [p, e] = await Promise.all([predict("churn", feats), explain("churn", feats)]);
      setPred(p);
      setExpl(e);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }, []);

  // Debounced re-score as sliders move.
  useEffect(() => {
    if (timer.current) clearTimeout(timer.current);
    timer.current = setTimeout(() => void run(values), 220);
    return () => {
      if (timer.current) clearTimeout(timer.current);
    };
  }, [values, run]);

  const groups = useMemo(() => {
    const order: FeatureSpec["group"][] = ["usage", "support", "plan"];
    const names: Record<FeatureSpec["group"], string> = {
      usage: "Usage",
      support: "Account & support",
      plan: "Plans",
    };
    return order.map((g) => ({ g, name: names[g], items: FEATURES.filter((f) => f.group === g) }));
  }, []);

  const score = pred?.score ?? 0;
  const color = riskColor(score);

  return (
    <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_400px]">
      {/* --- controls ------------------------------------------------------- */}
      <div className="rounded-2xl border border-line bg-void-2 p-6 shadow-sm">
        <div className="flex items-center justify-between">
          <h3 className="text-[15px] font-semibold tracking-tight">Customer profile</h3>
          <div className="flex gap-2">
            <button
              onClick={() => setValues(DEFAULTS)}
              className="rounded-md border border-line px-2.5 py-1 text-[12px] text-ink-dim transition hover:bg-white/5"
            >
              Reset
            </button>
            <button
              onClick={() => setValues(ANGRY)}
              className="rounded-md bg-churn px-2.5 py-1 text-[12px] font-medium text-white transition hover:brightness-110"
            >
              Load at-risk profile
            </button>
          </div>
        </div>

        {groups.map(({ g, name, items }) => (
          <div key={g} className="mt-6">
            <div className="mb-3 font-mono text-[10px] uppercase tracking-[0.18em] text-ink-faint">
              {name}
            </div>
            <div className="space-y-3.5">
              {items.map((f) => (
                <div key={f.key} className="flex items-center gap-4">
                  <label className="w-44 shrink-0 text-[12.5px] text-ink-dim">{f.label}</label>
                  <input
                    type="range"
                    min={f.min}
                    max={f.max}
                    step={f.step}
                    value={values[f.key]}
                    onChange={(e) => setValues((v) => ({ ...v, [f.key]: Number(e.target.value) }))}
                    className="h-1.5 flex-1 cursor-pointer appearance-none rounded-full bg-line accent-churn"
                  />
                  <span className="w-14 shrink-0 text-right font-mono text-[12px] tabular-nums text-ink">
                    {f.max <= 1 ? (values[f.key] ? "yes" : "no") : values[f.key].toFixed(f.step < 1 ? 1 : 0)}
                  </span>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>

      {/* --- result --------------------------------------------------------- */}
      <div className="space-y-4">
        <div className="rounded-2xl border border-line bg-void-2 p-6 shadow-sm">
          <div className="font-mono text-[10px] uppercase tracking-[0.18em] text-ink-faint">
            Churn probability
          </div>

          <div className="mt-3 flex items-end gap-3">
            <span className="text-5xl font-semibold tabular-nums tracking-tight" style={{ color }}>
              {busy ? "—" : `${(score * 100).toFixed(1)}%`}
            </span>
            <span className="pb-1.5 text-[13px] font-medium" style={{ color }}>
              {pred?.label ?? ""}
            </span>
          </div>

          <div className="mt-4 h-2 overflow-hidden rounded-full bg-void-3">
            <motion.div
              className="h-full rounded-full"
              style={{ background: color }}
              animate={{ width: `${Math.max(2, score * 100)}%` }}
              transition={{ type: "spring", stiffness: 140, damping: 20 }}
            />
          </div>

          {pred && (
            <div className="mt-4 flex items-center justify-between font-mono text-[11px] text-ink-faint">
              <span>
                base rate {(Number(pred.meta.base_rate) * 100).toFixed(1)}%
              </span>
              <span>threshold {(Number(pred.meta.threshold) * 100).toFixed(0)}%</span>
            </div>
          )}

          {pred && (
            <div className="mt-5 rounded-xl bg-void-3 p-4 ring-1 ring-line">
              <div className="font-mono text-[10px] uppercase tracking-[0.18em] text-churn">
                Recommended action
              </div>
              <p className="mt-1.5 text-[13px] leading-relaxed text-ink-dim">{pred.decision}</p>
            </div>
          )}

          {busy && <p role="status" className="mt-3 text-[11px] text-ink-faint">Scoring this profile and preparing its explanation…</p>}
          {error && <div className="mt-4"><WakeRetry error={error} onRetry={() => void run(values)} busy={busy} label="Retry score" /></div>}
        </div>

        {/* --- explanation --------------------------------------------------- */}
        <div className="rounded-2xl border border-line bg-void-2 p-6 shadow-sm">
          <div className="flex items-baseline justify-between">
            <h3 className="text-[15px] font-semibold tracking-tight">Why this score</h3>
            {expl && (
              <span className="font-mono text-[10px] uppercase tracking-widest text-ink-faint">
                {expl.kind} · {expl.space}
              </span>
            )}
          </div>

          {expl && <p className="mt-2 text-[12.5px] text-ink-dim">{expl.summary}</p>}

          <div className="mt-5 space-y-2.5">
            {(expl?.attributions ?? []).slice(0, 8).map((a, i) => {
              const max = Math.max(...(expl?.attributions ?? []).map((x) => Math.abs(x.attribution)), 1e-6);
              const pct = (Math.abs(a.attribution) / max) * 100;
              const up = a.direction === "increases_risk";
              return (
                <motion.div
                  key={a.feature}
                  initial={{ opacity: 0, x: -6 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: i * 0.02 }}
                  className="flex items-center gap-3"
                >
                  <span className="w-36 shrink-0 truncate font-mono text-[10.5px] text-ink-dim">
                    {a.feature}
                  </span>
                  <div className="relative h-4 flex-1 rounded bg-void-3">
                    {/* Diverging bar from centre: right pushes risk up, left pushes it down. */}
                    <div className="absolute left-1/2 top-0 h-full w-px bg-line" />
                    <div
                      className="absolute top-0.5 h-3 rounded-sm"
                      style={{
                        background: up ? "var(--color-churn)" : "var(--color-ink-faint)",
                        width: `${pct / 2}%`,
                        left: up ? "50%" : `${50 - pct / 2}%`,
                      }}
                    />
                  </div>
                  <span className="w-14 shrink-0 text-right font-mono text-[10.5px] tabular-nums text-ink-dim">
                    {a.attribution > 0 ? "+" : ""}
                    {a.attribution.toFixed(2)}
                  </span>
                </motion.div>
              );
            })}
          </div>

          {expl && (
            <ul className="mt-5 space-y-1.5 border-t border-line pt-4">
              {expl.caveats.map((c) => (
                <li key={c} className="text-[11px] leading-relaxed text-ink-faint">
                  — {c}
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>
    </div>
  );
}
