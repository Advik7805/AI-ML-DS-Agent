# Nexus — an AI/ML/Data-Science agent platform

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

**Phase 0 shipped and running.** The contract, the gateway, the hub and the first body are
built, tested and verified end-to-end.

| Piece | State |
|---|---|
| Module contract + registry (`core/`) | 12 conformance rules, enforced in CI |
| Generic API gateway (`apps/api/`) | zero module-specific code |
| Churn body (`modules/churn/`) | live, on the real 3,333-row telecom dataset |
| Hub with 3D module graph (`apps/web/app/(hub)/`) | live |
| Churn storefront (`apps/web/app/(churn)/`) | live |
| 5 further bodies | roadmap — see §Bodies |

**50 tests passing.** Held-out results on the real dataset, read from
`GET /api/v1/modules/churn/metrics` rather than a notebook:

| Metric | Value |
|---|---|
| PR-AUC | **0.677** vs 0.145 baseline → **4.67× lift** |
| ROC-AUC | 0.856 |
| Threshold | 0.41, chosen by F1 on a fold reserved for it |
| Test confusion | tn 674 · fp 39 · fn 47 · tp 74 |

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
core/            the contract, shared types, module registry
modules/
  churn/         ← a body: data.py, train.py, service.py, tests. Self-contained.
apps/
  api/           FastAPI gateway. Generic — knows nothing about churn.
  web/           Next.js: (hub)/ and (churn)/ each own a root layout
data/fetch.sh    pulls the real datasets; git-ignored so the repo stays light
tests/           contract conformance, model quality, gateway over HTTP
docs/            RESEARCH · ARCHITECTURE · BUILD-NOTES
deploy/          API Dockerfile
```

---

## Run it

**Backend**

```bash
pip install -r requirements.txt
bash data/fetch.sh                    # real dataset; optional — synthetic fallback works
python -m pytest                      # 50 tests
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

| Body | State | Stack |
|---|---|---|
| **Churn & Retention** | **live** | GBM + SMOTE, sigmoid calibration, TreeSHAP |
| Knowledge RAG | planned | LlamaIndex, pgvector on Supabase |
| Fraud & Anomaly | planned | Redis Streams, SMOTE, live alerts |
| Predictive Maintenance | planned | PyTorch LSTM, sensor telemetry |
| Explainable Diagnostics | planned | PyTorch + SHAP + Fairlearn bias audits |
| Dynamic Pricing | planned | H3 spatial indexing, demand allocation |
| Scraper | planned | Firecrawl API, zero self-hosting |

---

## Deploying free

| Component | Target | Constraint designed around |
|---|---|---|
| Frontend | Vercel Hobby | — |
| API | Render free | sleeps after 15 min → the UI degrades to a notice, never a blank page |
| API image | `deploy/Dockerfile.api` | 512 MB RAM → no torch in this image |
| DB + vectors | Supabase free | ephemeral container disks are why vectors live in Postgres |
| Cache / streams | Upstash Redis free | real pub/sub without Kafka's RAM bill |
| LLM | Groq / Gemini free tiers | provider-swappable behind LangChain |
| CI | GitHub Actions | public repo, unlimited minutes |

---

## Documentation

| Doc | Contents |
|---|---|
| [`docs/RESEARCH.md`](docs/RESEARCH.md) | Every relevant repo, with verified star counts and activity dates. What to build on, what to read, what to avoid. |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | The contract, the storefront topology (with the Next.js mechanism verified against source), the 3D brief, the phased plan. |
| [`docs/BUILD-NOTES.md`](docs/BUILD-NOTES.md) | Five real defects the test suite caught — calibration leakage, score flattening, a SHAP/explained-model mismatch, and more. |

---

## Licence

TBD.
