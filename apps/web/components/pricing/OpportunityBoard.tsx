"use client";

/**
 * Pricing opportunity board.
 *
 * The honest part of this panel is the "not identified" row. An elasticity you cannot estimate
 * is worse than no estimate at all, because it invites a price change justified by noise. SKUs
 * whose price barely moved in the history are called out as needing a price test rather than a
 * recommendation.
 */

import { motion } from "motion/react";
import { useEffect, useState } from "react";

import { clientFetch } from "@/lib/api";

interface Sku {
  sku: string;
  elasticity: number;
  identified: boolean;
  recommended_price: number;
  estimated_revenue_uplift: number;
  r2: number;
}

interface Opportunities {
  n_skus: number;
  n_not_identified: number;
  not_identified: string[];
  top: Sku[];
  note: string;
}

const ACCENT = "#059669";

export function OpportunityBoard() {
  const [data, setData] = useState<Opportunities | null>(null);
  const [busy, setBusy] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<Sku | null>(null);
  const [detail, setDetail] = useState<Record<string, unknown> | null>(null);

  useEffect(() => {
    clientFetch<{ result: Opportunities }>("/api/v1/agent/tools/list_skus_by_opportunity", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ arguments: {} }),
    })
      .then((r) => setData(r.result))
      .catch((e) => setError(e instanceof Error ? e.message : String(e)))
      .finally(() => setBusy(false));
  }, []);

  async function optimise(sku: string) {
    setSelected(data?.top.find((s) => s.sku === sku) ?? null);
    setDetail(null);
    try {
      const r = await clientFetch<{ result: Record<string, unknown> }>(
        "/api/v1/agent/tools/optimise_price",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ arguments: { sku } }),
        },
      );
      setDetail(r.result);
    } catch (e) {
      setDetail({ error: e instanceof Error ? e.message : String(e) });
    }
  }

  if (error)
    return (
      <div className="rounded-2xl border border-red-500/30 bg-red-500/10 p-6 text-[13px] text-red-300">
        Pricing unavailable: {error}
      </div>
    );

  if (!data)
    return (
      <div className="animate-pulse rounded-2xl border border-line bg-void-3 p-6 text-[13px] text-ink-faint">
        Estimating elasticities…
      </div>
    );

  const maxUplift = Math.max(...data.top.map((s) => s.estimated_revenue_uplift), 0.01);

  return (
    <div className="grid gap-4 lg:grid-cols-[1fr_320px]">
      <div className="rounded-2xl border border-line bg-void-3 p-6 shadow-sm">
        <h3 className="text-[15px] font-semibold tracking-tight">Revenue opportunity by SKU</h3>
        <p className="mt-1 text-[11.5px] text-ink-faint">
          {data.n_skus - data.n_not_identified} of {data.n_skus} SKUs have an identified
          elasticity. Click a row for the full recommendation.
        </p>

        <div className="mt-4 overflow-x-auto">
          <table className="w-full text-left text-[12.5px]">
            <thead>
              <tr className="border-b border-line font-mono text-[10px] uppercase tracking-wider text-ink-faint">
                <th className="py-2 pr-3">sku</th>
                <th className="py-2 pr-3 text-right">elasticity</th>
                <th className="py-2 pr-3 text-right">price to</th>
                <th className="py-2 pr-3">uplift</th>
                <th className="py-2 text-right">fit r²</th>
              </tr>
            </thead>
            <tbody>
              {data.top.map((s, i) => (
                <motion.tr
                  key={s.sku}
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: i * 0.02 }}
                  onClick={() => void optimise(s.sku)}
                  className="cursor-pointer border-b border-white/5 transition hover:bg-white/5"
                >
                  <td className="py-2 pr-3 font-mono text-[12px]">{s.sku}</td>
                  <td className="py-2 pr-3 text-right font-mono tabular-nums">{s.elasticity.toFixed(2)}</td>
                  <td className="py-2 pr-3 text-right font-mono tabular-nums">
                    ${s.recommended_price.toFixed(2)}
                  </td>
                  <td className="py-2 pr-3">
                    <div className="flex items-center gap-2">
                      <div className="relative h-1.5 w-24 overflow-hidden rounded-full bg-white/5">
                        <div
                          className="h-full rounded-full"
                          style={{
                            width: `${Math.max(2, (Math.abs(s.estimated_revenue_uplift) / maxUplift) * 100)}%`,
                            background: s.estimated_revenue_uplift >= 0 ? ACCENT : "#d97706",
                          }}
                        />
                      </div>
                      <span
                        className="font-mono text-[11px] tabular-nums"
                        style={{ color: s.estimated_revenue_uplift >= 0 ? ACCENT : "#d97706" }}
                      >
                        {s.estimated_revenue_uplift >= 0 ? "+" : ""}
                        {(s.estimated_revenue_uplift * 100).toFixed(1)}%
                      </span>
                    </div>
                  </td>
                  <td className="py-2 text-right font-mono text-[11px] tabular-nums text-ink-dim">
                    {s.r2.toFixed(2)}
                  </td>
                </motion.tr>
              ))}
            </tbody>
          </table>
        </div>

        {data.not_identified.length > 0 && (
          <div className="mt-4 rounded-xl border border-churn/30 bg-churn/10 p-4">
            <div className="text-[12.5px] font-medium text-churn">
              {data.n_not_identified} SKU(s) with no identified elasticity:{" "}
              <span className="font-mono">{data.not_identified.join(", ")}</span>
            </div>
            <p className="mt-1 text-[11.5px] leading-relaxed text-churn">{data.note}</p>
          </div>
        )}
      </div>

      <div className="rounded-2xl border border-line bg-void-3 p-5 shadow-sm">
        <div className="font-mono text-[10px] uppercase tracking-[0.16em] text-ink-faint">
          Recommendation
        </div>

        {!selected && (
          <p className="mt-3 text-[12.5px] text-ink-faint">Select a SKU to see its elasticity
            confidence interval and the demand floor the recommendation respects.</p>
        )}

        {selected && !detail && (
          <p className="mt-3 animate-pulse text-[12.5px] text-ink-faint">Optimising…</p>
        )}

        {selected && detail && (
          <Detail meta={(detail.meta ?? {}) as Record<string, unknown>}
                  price={(detail.meta as Record<string, unknown>)?.recommended_price}
                  label={detail.label} decision={detail.decision} />
        )}
      </div>
    </div>
  );
}

const DETAIL_KEYS = ["elasticity", "elasticity_ci", "r2", "n_obs", "observed_price_range", "demand_floor"];

function Detail({ meta, price, label, decision }: {
  meta: Record<string, unknown>;
  price: unknown;
  label: unknown;
  decision: unknown;
}) {
  return (
    <div className="mt-3 space-y-3">
      <div className="text-2xl font-semibold tabular-nums tracking-tight" style={{ color: ACCENT }}>
        ${typeof price === "number" ? price.toFixed(2) : "—"}
      </div>
      <div className="text-[13px] font-medium">{String(label ?? "")}</div>
      <p className="text-[12px] leading-relaxed text-ink-dim">{String(decision ?? "")}</p>

      <dl className="space-y-1.5 border-t border-white/5 pt-3 text-[12px]">
        {Object.entries(meta)
          .filter(([k]) => DETAIL_KEYS.includes(k))
          .map(([k, v]) => (
          <div key={k} className="flex items-baseline justify-between gap-2">
            <dt className="font-mono text-[10.5px] text-ink-dim">{k.replace(/_/g, " ")}</dt>
            <dd className="text-right font-mono text-[11px] tabular-nums text-ink">
              {Array.isArray(v)
                ? `[${v.map((x) => (typeof x === "number" ? x.toFixed(3) : String(x))).join(", ")}]`
                : typeof v === "number"
                  ? v.toFixed(3)
                  : String(v)}
            </dd>
          </div>
        ))}
      </dl>

      <p className="border-t border-white/5 pt-3 text-[11px] leading-relaxed text-ink-faint">
        Recommendations are clipped to the price range actually observed in the history. An
        optimum outside that range would be an extrapolation, not a finding.
      </p>
    </div>
  );
}
