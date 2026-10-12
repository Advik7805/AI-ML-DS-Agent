"use client";

import { useSyncExternalStore } from "react";

/**
 * One browser session id, shared by everything that needs to remember something across requests.
 *
 * Why it exists: an uploaded file lives in a session on the API (never on disk), so *every*
 * request that should see it has to carry the same id. The upload studio and the hub agent chat
 * are separate components, and when they minted their own ids the result was the worst possible
 * behaviour — you upload a file, ask the agent about it in the chat box right underneath, and the
 * agent answers as if the file did not exist, because as far as that request was concerned it
 * didn't.
 *
 * So the key and the minting live here, in one place, and both components use it.
 */

export const SESSION_KEY = "saros.session";

/** Mint an id in the shape the API expects (`s_` + 20 chars). */
function mint(): string {
  try {
    return `s_${crypto.randomUUID().replace(/-/g, "").slice(0, 20)}`;
  } catch {
    return `s_${Math.random().toString(36).slice(2, 12)}${Date.now().toString(36).slice(-8)}`;
  }
}

/**
 * Read the session id, or mint and store one. Safe to call from an event handler or a request
 * header path; returns "" on the server, where there is no storage and no session to speak of.
 */
export function readSessionId(): string {
  if (typeof window === "undefined") return "";
  try {
    const existing = window.localStorage.getItem(SESSION_KEY);
    if (existing) return existing;
    const created = mint();
    window.localStorage.setItem(SESSION_KEY, created);
    return created;
  } catch {
    // Private mode with storage disabled: an ephemeral id still works for this page view.
    return mint();
  }
}

let cached: string | null = null;

/** Stable snapshot for useSyncExternalStore — a new id per render would loop forever. */
function snapshot(): string {
  if (cached === null) cached = readSessionId();
  return cached;
}

function subscribe(onChange: () => void): () => void {
  // Another tab (or the same page after a reset) changing the id should be picked up.
  window.addEventListener("storage", onChange);
  window.addEventListener("saros:session", onChange);
  return () => {
    window.removeEventListener("storage", onChange);
    window.removeEventListener("saros:session", onChange);
  };
}

/**
 * The session id as React state.
 *
 * `useSyncExternalStore` rather than `useState` + `useEffect` on purpose: the server snapshot is
 * empty, so nothing session-specific is rendered into the HTML, and the client picks the real id
 * up immediately after hydration without a mismatch warning.
 */
export function useSessionId(): string {
  return useSyncExternalStore(subscribe, snapshot, () => "");
}
