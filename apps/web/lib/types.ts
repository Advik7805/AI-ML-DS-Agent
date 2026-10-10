/**
 * TypeScript mirror of `core/types.py`.
 *
 * Kept hand-written and deliberately narrow: the Python side is the source of truth and the
 * OpenAPI schema at `/api/openapi.json` is generated from it. If these two drift, the contract
 * is broken and `tests/test_contract.py` is the place that should catch it.
 */

export type Direction = "increases_risk" | "decreases_risk" | "neutral";

/** Units the attributions live in. See `core.types.AttributionSpace`. */
export type AttributionSpace = "log_odds" | "probability" | "logits" | "unspecified";

export type ExplanationKind = "shap" | "attention" | "retrieval" | "heuristic" | "none";

export interface FeatureAttribution {
  feature: string;
  value: number;
  /** Signed contribution in `space` units — NOT necessarily probability. */
  attribution: number;
  direction: Direction;
}

export interface Explanation {
  kind: ExplanationKind;
  space: AttributionSpace;
  attributions: FeatureAttribution[];
  base_value: number | null;
  summary: string;
  caveats: string[];
}

export interface Prediction {
  module_id: string;
  score: number;
  label: string;
  decision: string;
  inputs: Record<string, number>;
  factors: FeatureAttribution[];
  explanation: Explanation | null;
  meta: Record<string, unknown>;
}

export interface AgentTool {
  name: string;
  description: string;
  parameters_schema: Record<string, unknown>;
  module_id: string;
}

export interface ModuleManifest {
  id: string;
  title: string;
  summary: string;
  accent: string;
  version: string;
  status: "live" | "beta" | "planned";
  capabilities: string[];
  routes: Record<string, string>;
  tools: AgentTool[];
}

export interface MetricsReport {
  n_samples: number;
  positive_rate: number;
  scores: Record<string, number>;
  confusion: Record<string, number>;
  threshold: number;
  notes: string[];
  /** Per-subpopulation breakdown (fairness, segments, operating regimes). Empty if none. */
  groups: Record<string, Record<string, number>>;
  /** How `groups` was sliced. Never empty when `groups` is populated. */
  group_basis: string;
}

export interface HealthResponse {
  status: string;
  modules_loaded: string[];
  load_errors: Record<string, string>;
}

/** The six bodies on the roadmap. Only `churn` is `live` today; the rest render as planned. */
export const ROADMAP: Array<{ id: string; title: string; blurb: string; accent: string }> = [
  {
    id: "rag",
    title: "Knowledge RAG",
    blurb: "Retrieval-augmented assistant over an enterprise corpus, with pgvector and cited answers.",
    accent: "#38bdf8",
  },
  {
    id: "fraud",
    title: "Fraud & Anomaly",
    blurb: "Streaming transaction scoring over Redis Streams, SMOTE for the minority class, live alerts.",
    accent: "#f43f5e",
  },
  {
    id: "maintenance",
    title: "Predictive Maintenance",
    blurb: "LSTM and gradient-boosted survival models on sensor telemetry to forecast failures.",
    accent: "#22d3ee",
  },
  {
    id: "xai-health",
    title: "Explainable Diagnostics",
    blurb: "PyTorch diagnostics with SHAP and Fairlearn bias audits for high-stakes decisions.",
    accent: "#a78bfa",
  },
  {
    id: "pricing",
    title: "Dynamic Pricing",
    blurb: "Demand allocation over POS and telemetry streams, with H3 spatial routing.",
    accent: "#34d399",
  },
];
