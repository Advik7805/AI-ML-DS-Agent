/**
 * Waking-aware fetching for a free-tier backend.
 *
 * The API sleeps after 15 minutes of inactivity. Every board that treated that as an error
 * showed a red box to a user who had done nothing wrong, when the correct answer was "wait
 * eight seconds". This hook makes that distinction explicit: a network-level failure is
 * `waking` and is retried with backoff, a 4xx/5xx body from a live API is `error`.
 *
 * `server` selects the transport: false (default) uses relative URLs through the Next rewrite,
 * true fetches the API origin directly the way a server component does.
 */

"use client";

import { useCallback, useEffect, useRef, useState } from "react";

export type WakeState = "idle" | "loading" | "waking" | "ready" | "error";

export interface WakeResult<T> {
  data: T | null;
  state: WakeState;
  error: string | null;
  /** Which attempt is in flight, for "retrying (2/3)" copy. */
  attempt: number;
  retry: () => void;
}

/** Backoff between retries. Three tries over ~10s covers a Render cold start. */
export const WAKE_BACKOFF_MS = [1200, 3500, 7000];

export function useWakingFetch<T>(path: string | null, options?: { retries?: number }): WakeResult<T> {
  const retries = options?.retries ?? WAKE_BACKOFF_MS.length;
  const [data, setData] = useState<T | null>(null);
  const [state, setState] = useState<WakeState>("idle");
  const [error, setError] = useState<string | null>(null);
  const [attempt, setAttempt] = useState(0);
  const [nonce, setNonce] = useState(0);
  const alive = useRef(true);

  useEffect(() => {
    alive.current = true;
    return () => {
      alive.current = false;
    };
  }, []);

  useEffect(() => {
    if (!path) return;
    let cancelled = false;
    const run = async () => {
      setState((s) => (s === "ready" ? s : "loading"));
      setError(null);
      for (let i = 0; i <= retries; i++) {
        if (cancelled) return;
        setAttempt(i + 1);
        try {
          const res = await fetch(path, { cache: "no-store" });
          if (!res.ok) {
            // The API answered. It is awake — this is a real error, not a cold start.
            let detail = `${res.status} ${res.statusText}`;
            try {
              const body = (await res.json()) as { detail?: unknown };
              if (typeof body.detail === "string") detail = body.detail;
            } catch {
              /* non-JSON error body */
            }
            if (!cancelled) {
              setError(detail);
              setState("error");
            }
            return;
          }
          const payload = (await res.json()) as T;
          if (!cancelled) {
            setData(payload);
            setError(null);
            setState("ready");
          }
          return;
        } catch (e) {
          // No HTTP response at all: unreachable or still booting. Wait and try again.
          if (cancelled) return;
          if (i === retries) {
            setError(
              e instanceof Error
                ? `${e.message} — the API did not come back after ${retries + 1} attempts.`
                : String(e),
            );
            setState("error");
            return;
          }
          setState("waking");
          await new Promise((r) => setTimeout(r, WAKE_BACKOFF_MS[Math.min(i, WAKE_BACKOFF_MS.length - 1)]));
        }
      }
    };
    void run();
    return () => {
      cancelled = true;
    };
  }, [path, retries, nonce]);

  const retry = useCallback(() => setNonce((n) => n + 1), []);

  return { data, state, error, attempt, retry };
}
