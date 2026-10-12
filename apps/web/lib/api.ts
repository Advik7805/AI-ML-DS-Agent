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

export async function clientFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, { ...init, cache: "no-store" });
  return parse<T>(res);
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
