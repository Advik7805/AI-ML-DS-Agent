/**
 * API client.
 *
 * Two entry points on purpose:
 *
 * - `serverFetch` runs in a React Server Component and talks to `API_ORIGIN` directly
 *   (the FastAPI container). Used for above-the-fold data so the first paint is not blocked
 *   on a client round-trip.
 * - `clientFetch` runs in the browser and uses **relative** URLs, which `next.config.ts`
 *   rewrites to the backend. The browser never learns the backend's address — that is what
 *   makes the sandbox preview host work, and it removes CORS from the picture entirely.
 */

import type { DatasetCard, Explanation, MetricsReport, ModuleManifest, Prediction, Tenant } from "./types";

const API_ORIGIN = process.env.API_ORIGIN ?? "http://127.0.0.1:8000";

async function parse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = (await res.json()) as { detail?: string };
      if (body.detail) detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* non-JSON error body */
    }
    throw new Error(detail);
  }
  return (await res.json()) as T;
}

export async function serverFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_ORIGIN}${path}`, {
    ...init,
    // Every read is fresh; the backend is the cache of record for model output.
    cache: "no-store",
  });
  return parse<T>(res);
}

/**
 * Waking-aware client fetch.
 *
 * A free-tier API sleeps after 15 minutes idle, so the first call after a nap *fails at the
 * transport layer* and then succeeds about ten seconds later. Every board used to render that
 * as a red error box, which is wrong twice over: the user did nothing wrong, and the correct
 * guidance is "wait". This retries with backoff and announces itself through window events that
 * a single `WakeIndicator` (mounted in each layout) renders once, instead of eleven components
 * each inventing their own wording.
 *
 * A 4xx/5xx response is *not* retried: the API answered, so it is awake and the error is real.
 */
export const WAKE_EVENT = "saros:wake";
export const AWAKE_EVENT = "saros:awake";
const WAKE_BACKOFF_MS = [1200, 3500, 7000];

function announce(event: string, detail?: unknown) {
  if (typeof window === "undefined") return;
  window.dispatchEvent(new CustomEvent(event, { detail }));
}

export async function clientFetch<T>(
  path: string,
  init?: RequestInit,
  options?: { retries?: number; label?: string },
): Promise<T> {
  const retries = options?.retries ?? WAKE_BACKOFF_MS.length;
  let lastError: unknown = null;

  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      const res = await fetch(path, { ...init, cache: "no-store" });
      if (attempt > 0) announce(AWAKE_EVENT);
      return await parse<T>(res);
    } catch (e) {
      // Distinguish "no HTTP response" (asleep / offline) from "the API said no".
      const isHttpError = e instanceof Error && /^\d{3} /.test(e.message);
      if (isHttpError || attempt === retries) {
        if (attempt > 0) announce(AWAKE_EVENT);
        throw e;
      }
      lastError = e;
      announce(WAKE_EVENT, { attempt: attempt + 1, retries, label: options?.label ?? path });
      await new Promise((r) => setTimeout(r, WAKE_BACKOFF_MS[Math.min(attempt, WAKE_BACKOFF_MS.length - 1)]));
    }
  }
  throw lastError instanceof Error ? lastError : new Error(String(lastError));
}

// --- typed helpers -----------------------------------------------------------

export const getModules = (server = true) =>
  (server ? serverFetch : clientFetch)<ModuleManifest[]>("/api/v1/modules");

export const getModule = (id: string, server = true) =>
  (server ? serverFetch : clientFetch)<ModuleManifest>(`/api/v1/modules/${id}`);

export const getMetrics = (id: string, server = true) =>
  (server ? serverFetch : clientFetch)<MetricsReport>(`/api/v1/modules/${id}/metrics`);

export const getDataset = (id: string, server = true) =>
  (server ? serverFetch : clientFetch)<DatasetCard>(`/api/v1/modules/${id}/dataset`);

export interface TenantResponse {
  tenant: Tenant;
  bodies: Array<{ id: string; dataset: DatasetCard }>;
}

export const getTenant = (server = true) =>
  (server ? serverFetch : clientFetch)<TenantResponse>("/api/v1/tenant");

export async function predict(
  id: string,
  features: Record<string, number>,
  server = false,
): Promise<Prediction> {
  return (server ? serverFetch : clientFetch)<Prediction>(`/api/v1/modules/${id}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ features }),
  });
}

export async function explain(
  id: string,
  features: Record<string, number>,
  server = false,
): Promise<Explanation> {
  return (server ? serverFetch : clientFetch)<Explanation>(`/api/v1/modules/${id}/explain`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ features }),
  });
}

export interface RankedCustomer {
  row_index: number;
  /** The tenant's shared key space — the same person appears in the fraud book under this id. */
  customer_id: string;
  state: string;
  score: number;
  label: string;
  top_driver: string | null;
  decision: string;
}

export interface RankedResponse {
  n_returned: number;
  n_scored: number;
  data_source: string;
  threshold: number;
  // Provenance travels with the list, so the table can say what it is showing.
  tenant?: string;
  business_unit?: string;
  dataset?: string;
  dataset_source?: string;
  dataset_origin?: string;
  dataset_rows?: number | null;
  join_key?: string;
  customers: RankedCustomer[];
}

export async function rankAtRisk(n = 10, minScore = 0): Promise<RankedResponse> {
  const res = await clientFetch<{ tool: string; module_id: string; result: RankedResponse }>(
    "/api/v1/agent/tools/rank_at_risk_customers",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ arguments: { n, min_score: minScore } }),
    },
  );
  return res.result;
}
