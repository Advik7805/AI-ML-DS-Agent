"use client";

/**
 * Integration status — what is live, what is mocked, and what a deploy still needs to set.
 *
 * This exists because "it works on my machine" and "it works deployed" differ by exactly this
 * list. A visitor who sees a search result cannot otherwise tell whether it came from Tavily or
 * from a deterministic placeholder, and an operator cannot tell which env vars are missing
 * without reading the source. Both are answered here, from the same endpoint the bodies use.
 *
 * It reports env var *names*, never values — the endpoint is public by design.
 */

import { motion } from "motion/react";
import { useEffect, useState } from "react";

import { clientFetch } from "@/lib/api";

interface Integration {
  id: string;
  name: string;
  purpose: string;
  enabled: boolean;
  has_key: boolean;
  key_env: string;
  enable_env: string;
  docs_url: string;
  free_tier: string;
  used_by: string[];
  options: Record<string, string>;
}

interface Summary {
  integrations: Integration[];
  enabled: string[];
  missing: Array<{ id: string; name: string; key_env: string; enable_env: string }>;
  note: string;
}

export function IntegrationsPanel() {
  const [data, setData] = useState<Summary | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    clientFetch<Summary>("/api/v1/integrations")
      .then(setData)
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
  }, []);

  if (error)
    return (
      <div className="glass rounded-2xl p-5 text-[12.5px] text-ink-faint">
        Integration status unavailable: {error}
      </div>
    );

  if (!data) return <div className="glass animate-pulse rounded-2xl p-5 text-[12.5px] text-ink-faint">Reading configuration…</div>;

  const live = data.integrations.filter((i) => i.enabled);
  const mocked = data.integrations.filter((i) => !i.enabled);

  return (
    <div className="glass rounded-2xl p-5">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <div className="flex items-center gap-2">
          <span className="relative flex h-2 w-2">
            <span
              className="absolute inline-flex h-full w-full animate-ping rounded-full opacity-60"
              style={{ background: live.length ? "var(--color-hub)" : "var(--color-churn)" }}
            />
            <span
              className="relative inline-flex h-2 w-2 rounded-full"
              style={{ background: live.length ? "var(--color-hub)" : "var(--color-churn)" }}
            />
          </span>
          <span className="text-[13px] font-semibold text-ink">External services</span>
        </div>
        <span className="font-mono text-[10px] uppercase tracking-widest text-ink-faint">
          {live.length} live · {mocked.length} mocked
        </span>
      </div>

      <p className="mt-2 text-[11.5px] leading-relaxed text-ink-faint">{data.note}</p>

      <div className="mt-4 grid gap-2 sm:grid-cols-2">
        {data.integrations.map((i, idx) => (
          <motion.div
            key={i.id}
            initial={{ opacity: 0, y: 4 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: idx * 0.03 }}
            className="rounded-xl border border-white/10 bg-white/[0.03] p-3.5"
          >
            <div className="flex items-start justify-between gap-2">
              <div className="min-w-0">
                <div className="truncate text-[12.5px] font-medium text-ink">{i.name}</div>
                <div className="mt-0.5 text-[11px] leading-snug text-ink-dim">{i.purpose}</div>
              </div>
              <span
                className="shrink-0 rounded-md px-1.5 py-0.5 font-mono text-[9.5px] uppercase tracking-wider"
                style={{
                  background: "var(--color-void-3)",
                  color: "var(--color-ink)",
                }}
              >
                {i.enabled ? "live" : "mock"}
              </span>
            </div>

            <div className="mt-2.5 space-y-1 border-t border-white/10 pt-2">
              <div className="font-mono text-[10px] leading-relaxed text-ink-faint">
                {i.key_env ? `${i.key_env} · ` : ""}
                {i.enable_env || "no flag"}
              </div>
              {i.free_tier && (
                <div className="text-[10.5px] leading-relaxed text-ink-faint">{i.free_tier}</div>
              )}
              {i.used_by.length > 0 && (
                <div className="flex flex-wrap gap-1 pt-0.5">
                  {i.used_by.map((u) => (
                    <span
                      key={u}
                      className="rounded bg-white/[0.06] px-1.5 py-0.5 font-mono text-[9.5px] text-ink-dim"
                    >
                      {u}
                    </span>
                  ))}
                </div>
              )}
            </div>
          </motion.div>
        ))}
      </div>

      {mocked.length > 0 && (
        <p className="mt-4 border-t border-white/10 pt-3 text-[11px] leading-relaxed text-ink-faint">
          Everything above marked <span className="font-mono text-churn">mock</span> returns
          deterministic placeholder data that labels itself, in the payload and in the UI. Nothing
          is silently faked — a result you cannot distinguish from a real one is worse than an
          error.
        </p>
      )}
    </div>
  );
}
