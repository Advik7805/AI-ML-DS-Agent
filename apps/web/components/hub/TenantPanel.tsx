"use client";

/**
 * The tenant panel — the one-screen answer to "so what is this, actually?".
 *
 * Every body in this platform belongs to the same fictional company and every dataset is named.
 * Putting that on the hub, rather than in a README, means a visitor never has to guess whether
 * the seven dashboards are related or whether the numbers appeared from nowhere.
 *
 * Fetched client-side with waking retries; when the API is asleep the mirrored facts render
 * with a note saying they are the mirror, so the section never goes blank or silently lies.
 */

import { useWakingFetch } from "@/lib/waking";
import { KNOWN_DATASETS, KNOWN_TENANT, type ModuleManifest, type Tenant } from "@/lib/types";

import { WakingNotice } from "../body/WakingNotice";

interface TenantResponse {
  tenant: Tenant;
  bodies: Array<{
    id: string;
    dataset: {
      business_unit: string;
      dataset: string;
      origin: keyof typeof ORIGIN_TONE;
      rows: number | null;
      key: string;
      target: string;
    };
  }>;
}

const ORIGIN_TONE: Record<string, string> = {
  "public-dataset": "#34d399",
  generated: "#f59e0b",
  corpus: "#38bdf8",
  uploaded: "#a78bfa",
  none: "#9aa1b8",
};

const ORIGIN_LABEL: Record<string, string> = {
  "public-dataset": "real public dataset",
  generated: "generated stand-in",
  corpus: "shipped documents",
  uploaded: "session upload",
  none: "live fetch",
};

export function TenantPanel({ modules }: { modules: ModuleManifest[] }) {
  const { data, state, error, attempt, retry } = useWakingFetch<TenantResponse>("/api/v1/tenant");

  const tenant = data?.tenant;
  const rows = data?.bodies ?? [];
  const titles = new Map(modules.map((m) => [m.id, m.title]));

  // One row per body: live card when the API answers, mirrored facts when it does not.
  const list = rows.length
    ? rows.map((b) => ({ id: b.id, ...b.dataset }))
    : Object.entries(KNOWN_DATASETS).map(([id, d]) => ({
        id,
        business_unit: d.unit,
        dataset: d.dataset,
        origin: d.origin,
        rows: d.rows,
        key: d.key,
        target: d.target,
      }));

  return (
    <div className="glass rounded-2xl p-6">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <div className="font-mono text-[10px] uppercase tracking-[0.22em] text-ink-faint">
            the tenant behind every body
          </div>
          <h3 className="mt-2 text-xl font-semibold tracking-tight">
            {tenant?.name ?? KNOWN_TENANT.name}
            <span className="ml-2 align-middle font-mono text-[10px] uppercase tracking-widest text-ink-faint">
              fictional · {tenant?.headcount ?? KNOWN_TENANT.headcount} people ·{" "}
              {tenant?.hq ?? KNOWN_TENANT.hq}
            </span>
          </h3>
        </div>
        <span className="rounded-full border border-emerald-500/40 px-2.5 py-1 font-mono text-[10px] uppercase tracking-wider text-emerald-300">
          1 company · 7 bodies · 1 contract
        </span>
      </div>

      <p className="mt-3 max-w-3xl text-[13px] leading-relaxed text-ink-dim">
        {tenant?.tagline ?? KNOWN_TENANT.tagline} Every dataset below is named with its row count
        and its true origin — the three marked <em>real public dataset</em> are citable public
        sets, the rest are generated and labelled as such. Nothing here is a mystery feed.
      </p>

      {state !== "ready" && (
        <div className="mt-4">
          <WakingNotice state={state} error={error} attempt={attempt} onRetry={retry} />
        </div>
      )}

      <div className="mt-5 overflow-x-auto">
        <table className="w-full min-w-[720px] border-collapse text-left">
          <thead>
            <tr className="border-b border-line font-mono text-[10px] uppercase tracking-wider text-ink-faint">
              <th className="pb-2 pr-4 font-normal">body</th>
              <th className="pb-2 pr-4 font-normal">business unit</th>
              <th className="pb-2 pr-4 font-normal">dataset</th>
              <th className="pb-2 pr-4 font-normal">origin</th>
              <th className="pb-2 pr-4 font-normal">rows</th>
              <th className="pb-2 font-normal">join key</th>
            </tr>
          </thead>
          <tbody>
            {list.map((row) => (
              <tr key={row.id} className="border-b border-line/60 last:border-0">
                <td className="py-2.5 pr-4 text-[12.5px] font-medium">
                  {titles.get(row.id) ?? row.id}
                </td>
                <td className="py-2.5 pr-4 text-[12.5px] text-ink-dim">{row.business_unit}</td>
                <td className="py-2.5 pr-4 text-[12.5px] text-ink-dim">{row.dataset}</td>
                <td className="py-2.5 pr-4">
                  <span
                    className="font-mono text-[10.5px]"
                    style={{ color: ORIGIN_TONE[row.origin] ?? "var(--color-ink-faint)" }}
                  >
                    {ORIGIN_LABEL[row.origin] ?? row.origin}
                  </span>
                </td>
                <td className="py-2.5 pr-4 font-mono text-[11.5px] tabular-nums text-ink-dim">
                  {row.rows ? row.rows.toLocaleString() : "—"}
                </td>
                <td className="py-2.5 font-mono text-[11px] text-ink-faint">{row.key}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <p className="mt-4 text-[11.5px] leading-relaxed text-ink-faint">
        Why a fictional company: three of these datasets are real and public (IBM Telco Customer
        Churn, NASA CMAPSS FD001, Pima Indians Diabetes) and four are generated. Attributing them
        to a named real corporation would be a fabrication, so the tenant is clearly fictional
        while every dataset keeps its true source. Each body page carries the full card —
        including what the data cannot answer.
      </p>
    </div>
  );
}
