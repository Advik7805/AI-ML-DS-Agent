"use client";

import { useEffect, useState } from "react";

/**
 * The shared "the backend is asleep, this is normal" banner.
 *
 * A free-tier API sleeps; that is a property of the deployment, not a bug in the user's
 * session. One component so every board says the same thing instead of seven variations of
 * red text — and so "waking" (retrying) never looks like "broken" (gave up).
 */
export function WakingNotice({
  state,
  error,
  attempt,
  retries,
  onRetry,
  accent,
}: {
  state: "idle" | "loading" | "waking" | "ready" | "error";
  error: string | null;
  attempt?: number;
  retries?: number;
  onRetry?: () => void;
  accent?: string;
}) {
  if (state === "ready" || state === "idle") return null;

  if (state === "loading") {
    return (
      <div className="rounded-xl border border-line bg-void-3/60 px-4 py-3 text-[12px] text-ink-dim">
        <span className="mr-2 inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-hub align-middle" />
        Loading…
      </div>
    );
  }

  if (state === "waking") {
    return (
      <div
        className="rounded-xl border px-4 py-3 text-[12px]"
        style={{
          borderColor: accent ? `${accent}55` : "var(--color-line)",
          background: "rgba(255,255,255,0.03)",
          color: "var(--color-ink-dim)",
        }}
      >
        <span className="mr-2 inline-block h-1.5 w-1.5 animate-pulse rounded-full align-middle" style={{ background: accent ?? "var(--color-hub)" }} />
        <strong className="font-medium text-ink">Waking the API…</strong> free-tier instances sleep
        after 15 minutes idle, so the first request can take ~10–30s.
        {attempt ? <span className="ml-1 text-ink-faint">(attempt {attempt}{retries ? `/${retries}` : ""})</span> : null}
      </div>
    );
  }

  return (
    <div className="rounded-xl border border-red-500/30 bg-red-500/10 px-4 py-3 text-[12px] text-red-200">
      <div className="flex items-start justify-between gap-3">
        <div>
          <strong className="font-medium">Could not reach the API.</strong>
          <div className="mt-1 text-red-200/80">{error}</div>
        </div>
        {onRetry && (
          <button
            onClick={onRetry}
            className="shrink-0 rounded-lg border border-red-400/40 px-2.5 py-1 text-[11.5px] font-medium text-red-100 transition hover:bg-red-500/20"
          >
            Retry
          </button>
        )}
      </div>
    </div>
  );
}

/**
 * One global indicator, driven by the events `clientFetch` emits.
 *
 * Mounted once per root layout rather than once per board: eleven components each showing their
 * own "retrying…" banner is noise, and eleven different wordings for the same situation is worse.
 */
export function WakeIndicator() {
  const [attempt, setAttempt] = useState<{ n: number; total: number } | null>(null);

  useEffect(() => {
    const onWake = (e: Event) => {
      const detail = (e as CustomEvent<{ attempt?: number; retries?: number }>).detail ?? {};
      setAttempt({ n: detail.attempt ?? 1, total: detail.retries ?? 3 });
    };
    const onAwake = () => setAttempt(null);
    window.addEventListener("saros:wake", onWake);
    window.addEventListener("saros:awake", onAwake);
    return () => {
      window.removeEventListener("saros:wake", onWake);
      window.removeEventListener("saros:awake", onAwake);
    };
  }, []);

  if (!attempt) return null;
  return (
    <div className="fixed bottom-4 left-1/2 z-50 -translate-x-1/2">
      <div className="glass flex items-center gap-2.5 rounded-full px-4 py-2 text-[12px] text-ink-dim shadow-2xl">
        <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-hub" />
        <span>
          <strong className="font-medium text-ink">Waking the API…</strong> free tiers sleep after
          15 min idle. Retrying ({attempt.n}/{attempt.total}).
        </span>
      </div>
    </div>
  );
}
