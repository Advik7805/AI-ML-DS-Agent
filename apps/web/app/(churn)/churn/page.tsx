import { getMetrics } from "@/lib/api";
import type { MetricsReport } from "@/lib/types";

import { ProvenanceCard } from "@/components/body/ProvenanceCard";
import { Portfolio } from "@/components/churn/Portfolio";
import { Scorer } from "@/components/churn/Scorer";

export const dynamic = "force-dynamic";

async function loadMetrics(): Promise<{ metrics: MetricsReport | null; error: string | null }> {
  try {
    return { metrics: await getMetrics("churn", true), error: null };
  } catch (err) {
    return { metrics: null, error: err instanceof Error ? err.message : String(err) };
  }
}

export default async function ChurnPage() {
  const { metrics, error } = await loadMetrics();
  const s = metrics?.scores ?? {};

  return (
    <div className="mx-auto max-w-6xl px-5">
      {/* --- header --------------------------------------------------------- */}
      <section className="pt-12 pb-8">
        <p className="font-mono text-[11px] uppercase tracking-[0.24em] text-[#d97706]">
          saros body · churn
        </p>
        <h1 className="mt-4 max-w-3xl text-4xl font-semibold leading-[1.1] tracking-tight text-ink sm:text-5xl">
          Score every customer&rsquo;s churn risk, and know exactly why.
        </h1>
        <p className="mt-5 max-w-2xl text-[15px] leading-relaxed text-ink-dim">
          <strong className="font-medium text-ink">Northwind Connect</strong> — the telecom arm of
          Northwind Global — runs 3.3m broadband and mobile subscriptions. This scores its subscriber
          book: gradient-boosted classification on the 3,333-row billing extract (the public IBM
          Telco Customer Churn set). SMOTE handles the{" "}
          {metrics ? `${(metrics.positive_rate * 100).toFixed(1)}%` : "~14.5%"} minority class, a
          sigmoid map calibrates the probabilities, the operating threshold is chosen by F1 on a
          reserved fold — and every prediction ships with its SHAP drivers and the customer_id the
          rest of the platform joins on.
        </p>
      </section>

      <section id="data" className="scroll-mt-24 pb-10">
        <ProvenanceCard moduleId="churn" accent="#f59e0b" />
      </section>

      {/* --- model card ----------------------------------------------------- */}
      {metrics ? (
        <section id="model" className="scroll-mt-24 pb-10">
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            <Stat label="PR-AUC" value={s.pr_auc} hint={`baseline ${s.baseline_pr_auc?.toFixed(3)}`} strong />
            <Stat label="ROC-AUC" value={s.roc_auc} />
            <Stat label="Recall" value={s.recall} hint={`precision ${s.precision?.toFixed(3)}`} />
            <Stat label="Threshold" value={metrics.threshold} hint={`test n=${metrics.n_samples}`} />
          </div>

          <div className="mt-3 rounded-2xl border border-line bg-void-3 p-6 shadow-sm">
            <div className="flex flex-wrap items-baseline justify-between gap-3">
              <h3 className="text-[15px] font-semibold tracking-tight">Model card</h3>
              <span className="font-mono text-[10.5px] text-ink-faint">
                confusion · tn {metrics.confusion.tn} / fp {metrics.confusion.fp} / fn{" "}
                {metrics.confusion.fn} / tp {metrics.confusion.tp}
              </span>
            </div>

            <p className="mt-2 max-w-3xl text-[12.5px] leading-relaxed text-ink-dim">
              PR-AUC is the headline, not ROC-AUC: at a {`${(metrics.positive_rate * 100).toFixed(1)}%`}
              positive rate, ROC-AUC flatters almost any classifier. Lift over the
              always-positive baseline is{" "}
              <strong className="font-semibold text-ink">
                {(s.pr_auc! / s.baseline_pr_auc!).toFixed(2)}×
              </strong>
              .
            </p>

            <details className="mt-4">
              <summary className="cursor-pointer font-mono text-[11px] uppercase tracking-[0.14em] text-ink-faint hover:text-ink-dim">
                Methodology notes ({metrics.notes.length})
              </summary>
              <ul className="mt-3 space-y-2 border-t border-white/5 pt-3">
                {metrics.notes.map((n) => (
                  <li key={n} className="text-[12px] leading-relaxed text-ink-dim">
                    <span className="mr-2 text-[#d97706]">—</span>
                    {n}
                  </li>
                ))}
              </ul>
            </details>
          </div>
        </section>
      ) : (
        <section className="pb-10">
          <div className="rounded-2xl border border-churn/30 bg-churn/10 p-5 text-[13px] text-ink">
            <strong className="font-semibold">Model card unavailable.</strong> {error}
            <p className="mt-1.5 text-[12px]">
              The backend trains on first use; a sleeping free-tier instance takes a few seconds
              to wake. The scorer and portfolio below retry on their own.
            </p>
          </div>
        </section>
      )}

      {/* --- scorer --------------------------------------------------------- */}
      <section id="scorer" className="scroll-mt-24 pb-10">
        <Scorer />
      </section>

      {/* --- portfolio ------------------------------------------------------ */}
      <section id="portfolio" className="scroll-mt-24 pb-16">
        <Portfolio />
      </section>
    </div>
  );
}

function Stat({
  label,
  value,
  hint,
  strong,
}: {
  label: string;
  value: number | undefined;
  hint?: string;
  strong?: boolean;
}) {
  return (
    <div
      className={`rounded-2xl border p-5 shadow-sm ${
        strong ? "border-[#fcd34d] bg-churn/10" : "border-line bg-void-3"
      }`}
    >
      <div className="font-mono text-[10px] uppercase tracking-[0.16em] text-ink-faint">{label}</div>
      <div className="mt-2 text-3xl font-semibold tabular-nums tracking-tight text-ink">
        {value === undefined ? "—" : value.toFixed(3)}
      </div>
      {hint && <div className="mt-1 font-mono text-[10.5px] text-ink-faint">{hint}</div>}
    </div>
  );
}
