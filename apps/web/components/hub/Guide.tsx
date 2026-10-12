"use client";

/**
 * The in-app guide: one card per body, in plain language.
 *
 * Everything here comes from the body's own manifest — the summary it publishes, the file it
 * accepts, the dataset it scores. Nothing is hand-copied from a README, so a body that changes
 * what it does cannot leave a stale guide behind. The sample questions are clickable and prefill
 * the hub agent, because "what can I actually do with this?" is answered faster by trying it than
 * by reading it.
 */

import { useMemo } from "react";

import type { ModuleManifest } from "@/lib/types";

/** Concrete questions per body. Kept here rather than in the API: they are UI copy, not contract. */
const SAMPLES: Record<string, string[]> = {
  churn: [
    "Rank the top 5 customers most likely to churn",
    "What are the churn model metrics?",
    "An angry customer on an international plan has made 7 service calls — what is their risk and why?",
  ],
  fraud: [
    "How good is the fraud model?",
    "Score a 480 EUR transaction at 3am with 9 transactions in the last hour",
    "Has the fraud feed drifted?",
  ],
  maintenance: [
    "Which engines need maintenance first?",
    "How accurate is the remaining-life model, by band?",
    "Has fleet telemetry drifted from training?",
  ],
  pricing: [
    "List the SKUs with the biggest pricing opportunity",
    "What price maximises revenue for SKU-004?",
    "How confident is the elasticity estimate for the top SKU?",
  ],
  rag: [
    "What is the refund window?",
    "What happens to a transaction that is held for review?",
    "How do I upload my own policy document?",
  ],
  scraper: [
    "What acquisition backends are live right now?",
    "Render https://example.com and show me the text",
    "Which backend should I use for a JavaScript-heavy page?",
  ],
  xai_health: [
    "Audit the diabetes model for fairness",
    "Why does a 52-year-old patient with high glucose score as high risk?",
    "What is the false-negative rate at the chosen threshold?",
  ],
};

export function Guide({ modules }: { modules: ModuleManifest[] }) {
  const cards = useMemo(
    () =>
      modules.map((m) => ({
        id: m.id,
        title: m.title,
        accent: m.accent,
        unit: m.dataset?.business_unit ?? "",
        why: m.dataset?.why ?? m.summary,
        accepts: m.accepts || "No file upload — this body reads live sources.",
        origin: m.dataset?.origin ?? "none",
        tools: m.tools.map((t) => t.name),
        href: m.routes?.ui ?? `/${m.id}`,
        samples: SAMPLES[m.id] ?? [],
      })),
    [modules],
  );

  function prefill(question: string) {
    // A custom event rather than shared state: the agent lives in a different section of a
    // server-rendered page, and this keeps the two components independent.
    window.dispatchEvent(new CustomEvent("saros:ask", { detail: question }));
    document.getElementById("agent")?.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  return (
    <section id="guide" className="scroll-mt-20 border-t border-line px-6 py-16 sm:px-10">
      <div className="mx-auto max-w-6xl">
        <h2 className="text-2xl font-semibold tracking-tight">What you can do here</h2>
        <p className="mt-3 max-w-3xl text-[15px] leading-relaxed text-ink-dim">
          Seven bodies, one company. Each card says what the body decides, which file it takes, and
          three questions worth asking. Click a question and it lands in the agent below, which can
          call any of these bodies as a tool.
        </p>

        <div className="mt-8 grid gap-4 lg:grid-cols-2">
          {cards.map((c) => (
            <div
              key={c.id}
              className="glass rounded-2xl p-5"
              style={{ borderColor: `${c.accent}33` }}
            >
              <div className="flex items-start justify-between gap-3">
                <div>
                  <a href={c.href} className="text-[15px] font-semibold tracking-tight hover:underline">
                    {c.title}
                  </a>
                  <div className="mt-0.5 font-mono text-[10px] uppercase tracking-wider text-ink-faint">
                    {c.unit} · {c.origin.replace("-", " ")}
                  </div>
                </div>
                <a
                  href={c.href}
                  className="shrink-0 rounded-lg border border-line px-2.5 py-1 text-[11.5px] transition hover:bg-white/5"
                  style={{ color: c.accent }}
                >
                  open ↗
                </a>
              </div>

              <p className="mt-3 text-[12.5px] leading-relaxed text-ink-dim">{c.why}</p>

              <div className="mt-3 rounded-xl border border-line bg-void-2/60 px-3 py-2.5">
                <span className="font-mono text-[10px] uppercase tracking-wider text-ink-faint">
                  accepts ·{" "}
                </span>
                <span className="text-[11.5px] leading-relaxed text-ink-dim">{c.accepts}</span>
              </div>

              {c.samples.length > 0 && (
                <div className="mt-3">
                  <div className="font-mono text-[10px] uppercase tracking-wider text-ink-faint">
                    try asking
                  </div>
                  <div className="mt-1.5 flex flex-wrap gap-1.5">
                    {c.samples.map((s) => (
                      <button
                        key={s}
                        onClick={() => prefill(s)}
                        className="rounded-full border border-line px-2.5 py-1 text-left text-[11px] text-ink-dim transition hover:border-hub/50 hover:text-ink"
                      >
                        {s}
                      </button>
                    ))}
                  </div>
                </div>
              )}

              <div className="mt-3 flex flex-wrap gap-x-2.5 gap-y-1">
                {c.tools.map((t) => (
                  <span key={t} className="font-mono text-[10px] text-ink-faint">
                    {t}
                  </span>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
