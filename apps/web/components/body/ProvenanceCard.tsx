"use client";

/**
 * "Where did these numbers come from?" — answered on every body page.
 *
 * A dashboard that shows a churn score without saying whose subscribers, from which file, over
 * which period is a dashboard nobody can audit. This renders the DatasetCard the body's module
 * publishes: tenant, business unit, dataset, grain, join key, target, and — the part that
 * matters most — whether the numbers are a real public dataset or generated, stated in the same
 * breath as the row count rather than buried.
 *
 * Fetching happens client-side with waking-aware retries, so a sleeping API produces a
 * "waking" line and then the real card, never a blank panel. If the API never answers, the
 * mirrored facts keep the claim on screen and say that they are the mirror.
 */

import { useMemo } from "react";

import { useWakingFetch } from "@/lib/waking";
import { KNOWN_DATASETS, KNOWN_TENANT, type DatasetCard } from "@/lib/types";

import { WakingNotice } from "./WakingNotice";

const ORIGIN_COPY: Record<DatasetCard["origin"], { label: string; tone: string }> = {
  "public-dataset": { label: "real public dataset", tone: "#34d399" },
  generated: { label: "generated (stand-in)", tone: "#f59e0b" },
  corpus: { label: "shipped documents", tone: "#38bdf8" },
  uploaded: { label: "your upload (session only)", tone: "#a78bfa" },
  none: { label: "no dataset — live fetch", tone: "#9aa1b8" },
};

export function ProvenanceCard({ moduleId, accent }: { moduleId: string; accent: string }) {
  const { data, state, error, attempt, retry } = useWakingFetch<DatasetCard>(
    `/api/v1/modules/${moduleId}/dataset`,
  );
  const mirror = KNOWN_DATASETS[moduleId];
  const card = useMemo(() => data ?? null, [data]);

  const origin = ORIGIN_COPY[(card?.origin ?? mirror?.origin ?? "none") as DatasetCard["origin"]];
  const rows = card?.observed_rows ?? card?.rows ?? mirror?.rows ?? null;
  const basis = card?.observed_rows
    ? `observed${card.observed_basis ? ` (${card.observed_basis})` : ""}`
    : card?.rows
      ? "shipped"
      : "";
  const unit = card?.business_unit ?? mirror?.unit ?? "";
  const dataset = card?.dataset ?? mirror?.dataset ?? "—";

  return (
    <section id="data" className="scroll-mt-24">
      <div
        className="rounded-2xl border bg-void-3/60 p-5"
        style={{ borderColor: `${accent}33` }}
      >
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <div className="font-mono text-[10px] uppercase tracking-[0.2em] text-ink-faint">
              data provenance · tenant
            </div>
            <div className="mt-1.5 flex flex-wrap items-baseline gap-2">
              <span className="text-[15px] font-semibold tracking-tight">
                {card?.company ?? KNOWN_TENANT.name}
              </span>
              <span className="text-[12px] text-ink-faint">→</span>
              <span className="text-[13px] font-medium" style={{ color: accent }}>
                {unit}
              </span>
              {!card && <span className="text-[11px] text-ink-faint">(mirror — API asleep)</span>}
            </div>
          </div>
          <span
            className="rounded-full border px-2.5 py-1 font-mono text-[10px] uppercase tracking-wider"
            style={{ borderColor: `${origin.tone}55`, color: origin.tone }}
          >
            {origin.label}
          </span>
        </div>

        <div className="mt-4 grid gap-x-6 gap-y-3 sm:grid-cols-2 lg:grid-cols-3">
          <Field label="Dataset" value={dataset} />
          <Field
            label="Rows"
            value={
              rows
                ? `${rows.toLocaleString()}${basis ? ` · ${basis}` : ""}${
                    // The loader's own word for what it read — "real", "synthetic", "cmapss-fd001".
                    // Printed next to the row count because that is where a reader looks when
                    // deciding whether these numbers are the citable ones.
                    card?.observed_source ? ` · ${card.observed_source}` : ""
                  }`
                : "n/a"
            }
          />
          <Field label="Grain" value={card?.grain ?? "—"} />
          <Field label="Join key" value={card?.key || mirror?.key || "—"} />
          <Field label="Target" value={card?.target ?? mirror?.target ?? "—"} />
          <Field label="Period" value={card?.period ?? "—"} />
        </div>

        {state !== "ready" && (
          <div className="mt-4">
            <WakingNotice state={state} error={error} attempt={attempt} onRetry={retry} accent={accent} />
          </div>
        )}

        {card && (
          <>
            <p className="mt-4 text-[12.5px] leading-relaxed text-ink-dim">{card.origin_detail}</p>
            {card.origin_url && (
              <a
                href={card.origin_url}
                target="_blank"
                rel="noreferrer"
                className="mt-2 inline-block font-mono text-[11px] underline decoration-dotted"
                style={{ color: accent }}
              >
                source ↗
              </a>
            )}
            <div className="mt-4 grid gap-4 sm:grid-cols-2">
              <div className="rounded-xl border border-line bg-void-2/60 p-3.5">
                <div className="font-mono text-[10px] uppercase tracking-wider text-ink-faint">
                  file / feed
                </div>
                <p className="mt-1.5 text-[11.5px] leading-relaxed text-ink-dim">{card.file}</p>
                <div className="mt-2.5 font-mono text-[10px] uppercase tracking-wider text-ink-faint">
                  production system of record
                </div>
                <p className="mt-1.5 text-[11.5px] leading-relaxed text-ink-dim">
                  {card.system_of_record}
                </p>
              </div>
              <div className="rounded-xl border border-line bg-void-2/60 p-3.5">
                <div className="font-mono text-[10px] uppercase tracking-wider text-ink-faint">
                  why this model exists
                </div>
                <p className="mt-1.5 text-[11.5px] leading-relaxed text-ink-dim">{card.why}</p>
                <div className="mt-2.5 font-mono text-[10px] uppercase tracking-wider text-ink-faint">
                  how it connects to the other bodies
                </div>
                <p className="mt-1.5 text-[11.5px] leading-relaxed text-ink-dim">{card.connect}</p>
              </div>
            </div>
            {card.caveats.length > 0 && (
              <ul className="mt-4 space-y-1.5 border-t border-line pt-3.5">
                {card.caveats.map((c) => (
                  <li key={c} className="text-[11.5px] leading-relaxed text-ink-faint">
                    — {c}
                  </li>
                ))}
              </ul>
            )}
          </>
        )}
      </div>
    </section>
  );
}

function Field({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="font-mono text-[10px] uppercase tracking-wider text-ink-faint">{label}</div>
      <div className="mt-1 text-[12.5px] leading-snug text-ink">{value}</div>
    </div>
  );
}
