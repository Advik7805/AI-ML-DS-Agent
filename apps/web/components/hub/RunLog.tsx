"use client";

import { useCallback, useEffect, useState } from "react";

import { clientFetch } from "@/lib/api";

interface RunRecord {
  run_id: string;
  body: string;
  started_at: string;
  backend: string;
  metrics: Record<string, number>;
  params: Record<string, unknown>;
}

interface RunsResponse {
  status: {
    backend: string;
    mlflow_uri: string | null;
    runs_file: string;
    runs_file_entries: number;
    warning: string | null;
  };
  runs: RunRecord[];
}

/** The headline metric differs per body; show whichever this run actually reported. */
const HEADLINES: Record<string, string> = {
  churn: "pr_auc",
  fraud: "pr_auc",
  maintenance: "r2",
  xai_health: "roc_auc",
  pricing: "share_identified",
  rag: "hit@1",
};

/**
 * Experiment log. Seven bodies, each selecting an estimator by a metric, makes "which run
 * produced the model that is loaded" unanswerable without this.
 *
 * The backend is named in the panel rather than implied. On Render's free tier the filesystem is
 * ephemeral, so the local JSONL log is lost on every redeploy — the API says so, and hiding that
 * would make the panel look like durable history when it is not.
 */
export function RunLog() {
  const [data, setData] = useState<RunsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(true);

  const load = useCallback(() => {
    setBusy(true);
    clientFetch<RunsResponse>("/api/v1/runs?limit=12")
      .then(setData)
      .catch((e) => setError(e instanceof Error ? e.message : String(e)))
      .finally(() => setBusy(false));
  }, []);

  useEffect(load, [load]);

  if (error) {
    return (
      <div className="rounded-2xl border border-line bg-void-2/70 p-5 text-[12.5px] text-ink-dim">
        Run log unavailable ({error}). The backend may be asleep.
      </div>
    );
  }

  if (busy || !data) {
    return (
      <div className="animate-pulse rounded-2xl border border-line bg-void-2/70 p-5 text-[12.5px] text-ink-dim">
        Loading training history…
      </div>
    );
  }

  return (
    <div className="rounded-2xl border border-line bg-void-2/70 p-5">
      <div className="flex items-baseline justify-between gap-3">
        <div className="font-mono text-[10px] uppercase tracking-[0.16em] text-ink-dim">
          Training runs
        </div>
        <div className="font-mono text-[10px] text-ink-dim">
          {data.status.backend === "mlflow" ? "mlflow" : "local jsonl"} ·{" "}
          {data.status.runs_file_entries} recorded
        </div>
      </div>

      {data.runs.length === 0 ? (
        <p className="mt-3 text-[12.5px] leading-relaxed text-ink-dim">
          No runs recorded yet. Every body logs on <code>train()</code> — trigger one from a
          body&rsquo;s console, or run <code>python -m flows.train_all</code>.
        </p>
      ) : (
        <div className="mt-3 overflow-x-auto">
          <table className="w-full text-left font-mono text-[11px]">
            <thead>
              <tr className="text-[9.5px] uppercase tracking-[0.12em] text-ink-dim">
                <th className="pb-1.5 pr-3 font-medium">Body</th>
                <th className="pb-1.5 pr-3 font-medium">Headline</th>
                <th className="pb-1.5 pr-3 font-medium">Samples</th>
                <th className="pb-1.5 pr-3 font-medium">Seconds</th>
                <th className="pb-1.5 font-medium">When</th>
              </tr>
            </thead>
            <tbody>
              {data.runs.map((r) => {
                const key = HEADLINES[r.body];
                const headline =
                  key && typeof r.metrics?.[key] === "number"
                    ? `${key} ${r.metrics[key].toFixed(4)}`
                    : "—";
                return (
                  <tr key={r.run_id} className="border-t border-line/60">
                    <td className="py-1.5 pr-3 text-ink">{r.body}</td>
                    <td className="py-1.5 pr-3 text-ink">{headline}</td>
                    <td className="py-1.5 pr-3 tabular-nums text-ink-dim">
                      {r.metrics?.n_samples ?? "—"}
                    </td>
                    <td className="py-1.5 pr-3 tabular-nums text-ink-dim">
                      {typeof r.metrics?.train_seconds === "number"
                        ? r.metrics.train_seconds.toFixed(2)
                        : "—"}
                    </td>
                    <td className="py-1.5 text-ink-dim">{r.started_at}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {data.status.warning && (
        <p className="mt-3 border-t border-line/60 pt-2.5 text-[10.5px] leading-relaxed text-ink-dim">
          {data.status.warning}
        </p>
      )}
    </div>
  );
}
