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

/** Mirror of `core.types.DatasetCard` — where a body's numbers come from and whose they are. */
export interface DatasetCard {
  module_id: string;
  company: string;
  company_note: string;
  business_unit: string;
  dataset: string;
  grain: string;
  key: string;
  target: string;
  period: string;
  origin: "public-dataset" | "generated" | "corpus" | "uploaded" | "none";
  origin_detail: string;
  origin_url: string;
  file: string;
  rows: number | null;
  observed_rows: number | null;
  observed_source: string;
  observed_basis: string;
  system_of_record: string;
  why: string;
  connect: string;
  caveats: string[];
}

export interface Tenant {
  id: string;
  name: string;
  legal_name: string;
  tagline: string;
  founded: string;
  hq: string;
  headcount: string;
  revenue: string;
  why_fictional: string;
  units: Record<string, string>;
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
  /** What file this body accepts in the upload studio, in one sentence. */
  accepts: string;
  dataset: DatasetCard | null;
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

/**
 * Static mirror of the seven bodies. ONLY a fallback for when GET /api/v1/modules cannot be
 * reached — a free-tier backend sleeps, and the hub must still render something clickable.
 * When the API answers, its manifests win over every field here. If a body listed here and a
 * body in the API ever disagree, the API is right and this list is the one to update.
 */
export const KNOWN_BODIES: Array<{
  id: string;
  title: string;
  blurb: string;
  accent: string;
  href: string;
}> = [
  {
    id: "churn",
    title: "Churn & Retention",
    blurb:
      "Gradient-boosted churn scoring with SMOTE for class imbalance, isotonic calibration, and SHAP explanations. Ranks an entire book of business by attrition risk and recommends a retention action per customer.",
    accent: "#f59e0b",
    href: "/churn",
  },
  {
    id: "fraud",
    title: "Fraud & Anomaly",
    blurb:
      "Streaming transaction scoring. A supervised gradient-boosted model for known patterns plus an isolation forest for novel ones, combined by max so unseen fraud still trips the alarm. Live feed included.",
    accent: "#f43f5e",
    href: "/fraud",
  },
  {
    id: "maintenance",
    title: "Predictive Maintenance",
    blurb:
      "Remaining-useful-life regression on turbofan sensor telemetry, cross-validated by engine so the error is measured on hardware the model never saw. Error is reported per remaining-life band, because overall RMSE hides where it fails.",
    accent: "#22d3ee",
    href: "/maintenance",
  },
  {
    id: "pricing",
    title: "Dynamic Pricing",
    blurb:
      "Own-price elasticity per SKU from point-of-sale history, then revenue-optimal price subject to a demand floor. Recommendations are clipped to the observed price range and carry their confidence interval.",
    accent: "#34d399",
    href: "/pricing",
  },
  {
    id: "rag",
    title: "Knowledge RAG",
    blurb:
      "Retrieval-augmented answers over an enterprise policy corpus. Heading-aware chunking, TF-IDF ranking, and mandatory citations so nothing is asserted without a source.",
    accent: "#38bdf8",
    href: "/rag",
  },
  {
    id: "scraper",
    title: "Web Scraper",
    blurb:
      "Turns a URL into structured text the other bodies can consume. Host-allowlisted and size-capped so the agent cannot be talked into hammering a site. Mocked by default; Firecrawl is the production backend.",
    accent: "#f97316",
    href: "/scraper",
  },
  {
    id: "xai_health",
    title: "Explainable Diagnostics",
    blurb:
      "Diabetes risk scoring with per-patient SHAP explanations and a Fairlearn group fairness audit. Optimises for false-negative rate, because in screening a missed diagnosis is the harm that matters.",
    accent: "#a78bfa",
    href: "/diagnostics",
  },
];

/**
 * Offline provenance mirror.
 *
 * Same rule as KNOWN_BODIES: the API is the source of truth and wins whenever it answers, but
 * a free-tier backend sleeps — and a body page that goes blank, or worse, quietly drops the
 * "where did these numbers come from" panel, is the failure this project exists to avoid. So
 * the minimum set needed to keep the claim honest is mirrored here: who, what dataset, real or
 * generated, how many rows.
 */
export const KNOWN_TENANT = {
  name: "Northwind Global",
  tagline: "A fictional multinational, used as one consistent tenant across all seven bodies.",
  hq: "Rotterdam · Singapore · Austin",
  headcount: "48,000",
};

export const KNOWN_DATASETS: Record<
  string,
  { unit: string; dataset: string; origin: DatasetCard["origin"]; rows: number | null; key: string; target: string }
> = {
  churn: {
    unit: "Northwind Connect",
    dataset: "Subscriber billing & behaviour extract",
    origin: "public-dataset",
    rows: 3333,
    key: "customer_id",
    target: "churn within 30 days",
  },
  fraud: {
    unit: "Northwind Pay",
    dataset: "Card authorisation ledger",
    origin: "generated",
    rows: 20000,
    key: "auth_id + customer_id",
    target: "confirmed fraudulent authorisation",
  },
  maintenance: {
    unit: "Northwind Industrial",
    dataset: "Engine telemetry (NASA CMAPSS-shaped)",
    origin: "public-dataset",
    rows: 20631,
    key: "unit + cycle",
    target: "remaining useful life in cycles",
  },
  pricing: {
    unit: "Northwind Retail",
    dataset: "Point-of-sale demand history",
    origin: "generated",
    rows: 9600,
    key: "sku",
    target: "units sold (elasticity)",
  },
  rag: {
    unit: "Northwind Knowledge",
    dataset: "Group policy & procedure corpus",
    origin: "corpus",
    rows: null,
    key: "source file + heading",
    target: "cited policy answers",
  },
  scraper: {
    unit: "Northwind Labs",
    dataset: "Public web pages (competitive price intel)",
    origin: "none",
    rows: null,
    key: "url + timestamp",
    target: "acquired text for the other bodies",
  },
  xai_health: {
    unit: "Northwind Care",
    dataset: "Employee health screening pilot",
    origin: "public-dataset",
    rows: 768,
    key: "screening_id",
    target: "diabetes screening risk",
  },
};
