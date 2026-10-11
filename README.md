# Saros — an AI/ML/Data-Science agent platform

> One hub. Many independent bodies. Every body a real, working system.

A full-stack platform assembling the toolchain a senior data scientist actually uses into **one
connected product**: a multimodal agent hub that reasons across specialist bodies — RAG
knowledge assistants, real-time fraud and anomaly detection, predictive maintenance,
explainable AI for high-stakes decisions, dynamic pricing, churn and personalisation — plus
autonomous web scraping to feed them.

Each body is **its own storefront**: its own root layout, theme, route, API router, models and
tests. Clicking one takes you somewhere that genuinely looks and behaves like a different
site — not a tab in the same shop. All of them connect back to the hub through one shared
**module contract**, so the agent can call any of them and every body's predictions *and*
explanations render through the same components.

---

## Status

**All seven bodies built, discovered and agent-reachable.** Every body owns its own Next.js root
layout, so each route is a genuinely separate document — its own `<html>`, theme colour, metadata
and identity — while connecting to the hub through one shared contract.

| Body | Route | Data | Headline result |
|---|---|---|---|
| Churn & retention | `/churn` | real telecom, 834 held-out | PR-AUC **0.677** vs 0.145 → **4.67× lift** |
| Fraud & anomaly | `/fraud` | synthetic, 5,000 held-out | PR-AUC **0.608** vs 0.010 → **59.6× lift**, FPR 0.0002 |
| Predictive maintenance | `/maintenance` | synthetic CMAPSS-compatible | R² **0.840**, MAE 11.15 cycles (GroupKFold by engine) |
| Dynamic pricing | `/pricing` | synthetic point-of-sale | 22/24 SKUs with identified elasticity |
| Explainable diagnostics | `/diagnostics` | real Pima diabetes | ROC-AUC 0.690; fairness gap **0.208** flagged |
| Knowledge RAG | `/rag` | 6-doc policy corpus | hit@1 **0.833**, 25 passages |
| Web scraper | `/scraper` | mock by default | host-allowlisted, size-capped |

**176 tests passing** (`python3 -m pytest -q`), across six suites: contract conformance,
per-body model quality, the gateway over HTTP, integrations, the LLM providers, and the churn
body. **20 tools** register into one registry, names globally unique.

Results are read from the live gateway, not a notebook — e.g.
`GET /api/v1/modules/churn/metrics` and `GET /api/v1/agent/tools`.

The agent runs on a deterministic mock provider by default. Set `SAROS_LLM=groq` or
`SAROS_LLM=gemini` with the matching key for real reasoning; a missing key falls back to mock
rather than failing.

---

## Why this project

Research into the GitHub landscape (2026-10-10, measured via the Search API) found a split with
nothing in between:

- **Frameworks** are enormous and generic — `langchain` 147,559★, `firecrawl` 190,163★,
  `pytorch` 104,083★, `langgraph` 43,019★.
- **End products** are fragments — churn dashboards at 12–18★, IoT maintenance at 15–22★, fraud
  engines at 11–34★. Every one is a single module, standalone.

Searches for the *combination* returned nothing above ~22★. **The unified platform is the gap.**
The ML is commodity; integration quality and presentation are not.

Full findings, with verified star counts and last-push dates: [`docs/RESEARCH.md`](docs/RESEARCH.md)

---

## The idea that makes it tractable

Six bodies + an agent + a scraper + a 3D frontend is only buildable if **every body implements
one interface**. Then the hub can list and launch any body, the agent can call any body as a
tool, the frontend can render any body's output with shared components, and body #7 costs one
directory rather than a refactor.

```python
class Module(ABC):
    @property
    def manifest(self) -> ModuleManifest: ...      # how the hub renders and routes to it
    def predict(self, payload) -> Prediction: ...  # score one record
    def tools(self) -> list[AgentTool]: ...        # what the agent may call
    def explain(self, payload) -> Explanation: ... # optional, but this is the differentiator
```

Two rules keep it honest, both enforced in CI:

1. **A module may import `core`, never another module.** Verified by AST scan over every
   package under `modules/`. This is what makes the bodies independent rather than nominally so.
2. **Every declared tool must be dispatchable**, and tool names must be globally unique — the
   agent routes by bare name, so a collision would silently reach the wrong body.

---

## Layout

```
core/
  types.py       Prediction · Explanation · MetricsReport · ModuleManifest — one shape for all
  contract.py    the Module ABC and the registry that discovers bodies
  agent/         LangGraph plan-act loop; mock/groq/gemini providers
  integrations/  env-gated external services, each with an honest mock fallback
modules/
  churn/         ← a body: data.py, train.py, service.py. Self-contained.
  fraud/ maintenance/ pricing/ rag/ scraper/ xai_health/
apps/
  api/           FastAPI gateway. Generic — knows nothing about any one body.
  web/           Next.js: (hub)/ plus one root layout per body
data/fetch.sh    pulls the real datasets; refuses Git LFS pointers
tests/           contract conformance, per-body quality, integrations, gateway over HTTP
docs/            RESEARCH · ARCHITECTURE · BUILD-NOTES
deploy/          API Dockerfile
.env.example     every variable the platform understands, with its free-tier terms
```

---

## Run it

**Backend**

```bash
pip install -r requirements.txt
bash data/fetch.sh                    # real dataset; optional — synthetic fallback works
python -m pytest                      # 176 tests
python -m modules.churn.train         # held-out metrics
uvicorn apps.api.main:app --reload    # http://localhost:8000/docs
```

**Frontend**

```bash
cd apps/web && npm install && npm run dev    # http://localhost:3000
```

The browser only ever speaks to the Next.js origin; `next.config.ts` proxies `/api` to the
backend, which is what makes it work behind a preview host with no CORS.

---

## Bodies

| Body | State | Stack as actually built |
|---|---|---|
| **Churn & Retention** | live | Gradient boosting + SMOTE, sigmoid calibration over a `FrozenEstimator`, TreeSHAP |
| **Fraud & Anomaly** | live | GBM + SMOTE, Platt-calibrated Isolation Forest on the decision function |
| **Predictive Maintenance** | live | Gradient boosting over sensor windows, GroupKFold by engine, per-band MAE |
| **Explainable Diagnostics** | live | Gradient boosting on real Pima data, SHAP + Fairlearn disparity audit |
| **Knowledge RAG** | live | TF-IDF over a chunked markdown corpus; optional extra documents from Supabase |
| **Dynamic Pricing** | live | Log-log demand elasticity with standard errors, per-SKU opportunity ranking |
| **Data Acquisition** | live | Tavily search + extract, Firecrawl JS rendering, SSRF-guarded fetch |
| Scraper | planned | Firecrawl API, zero self-hosting |

---

## Deploying free

| Component | Target | Constraint designed around |
|---|---|---|
| Frontend | Vercel Hobby | — |
| API | Render free | sleeps after 15 min, ~60 s cold start → keep-alive cron + a "waking up" UI, never a blank page |
| API image | `deploy/Dockerfile.api` | 512 MB RAM → no torch in this image |
| DB + vectors | Supabase free | Render's free Postgres **expires after 30 days**, so nothing durable lives there |
| Cache / rate limits | Upstash Redis free | 10,000 commands/day is enough to stop one crawler exhausting the API |
| Web search / extraction | Tavily free | 1,000 credits/month, no card, students free |
| LLM | Groq / Gemini free tiers | provider-swappable behind one interface |
| CI | GitHub Actions | public repo, unlimited minutes |

**Why Render.** Checked in 2026: Railway removed its free tier in 2023 (now a one-time $5 trial
credit) and Fly.io ended free allowances for new accounts in 2024. Render is the only one of the
three with an ongoing free tier — 750 instance-hours/month, no card. Its cold start is the price,
and [`.github/workflows/keep-alive.yml`](.github/workflows/keep-alive.yml) pings `/health` every
10 minutes to stay inside the 15-minute idle window at zero cost.

---

## Optional integrations

Nothing here is required. With an empty environment the platform runs end to end on real and
generated data, and every external service degrades to a deterministic mock **that labels
itself** — in the API payload (`backend: "mock"`) and in the UI. An agent that cannot tell
simulated retrieval from real retrieval will confidently cite things that never happened.

Two rules, both enforced in code rather than convention:

1. **A key alone never enables a metered service.** Each also needs its `*_LIVE=1` flag, so a key
   pasted in for a local experiment cannot make a demo billable.
2. **Missing config is not an error.** `enabled` is a property the UI reads, so a storefront
   renders "connect Tavily to go live" instead of throwing.

`GET /api/v1/integrations` reports all of it — env var *names*, never values — so the frontend can
show what a deploy still needs. Copy [`.env.example`](.env.example) for the full list.

| Service | Used by | Free tier | Wired? |
|---|---|---|---|
| Tavily | scraper, rag | 1,000 credits/mo, no card | ✅ `web_search`, `extract_url` |
| Groq / Gemini | agent | free tiers | ✅ `SAROS_LLM=groq\|gemini` |
| Supabase | rag | 500 MB, free forever | ✅ `knowledge_documents` — extra corpus, no redeploy |
| Upstash | api, scraper | 10k commands/day | ✅ rate limiting on every metered route |
| Firecrawl | scraper | 500 credits one-time | ✅ `render_url` (JS-rendering) |


---

## Documentation

| Doc | Contents |
|---|---|
| [`docs/RESEARCH.md`](docs/RESEARCH.md) | Every relevant repo, with verified star counts and activity dates. What to build on, what to read, what to avoid. |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | The contract, the storefront topology (with the Next.js mechanism verified against source), the 3D brief, the phased plan. |
| [`docs/DEPLOY.md`](docs/DEPLOY.md) | The $0 runbook: Render blueprint, Vercel import, keep-alive, and which integrations to enable first. |
| [`docs/BUILD-NOTES.md`](docs/BUILD-NOTES.md) | Five real defects the test suite caught — calibration leakage, score flattening, a SHAP/explained-model mismatch, and more. |

---

## Licence

TBD.
