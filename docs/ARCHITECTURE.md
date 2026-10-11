# Architecture — "Saros": one hub, many independent bodies

Working name: **Saros** (rename freely). The idea in your words: *a clean site that has
everything, where each component is an independent body, and all bodies connect back to a
single hub.*

This document turns that into something buildable, and flags the decisions that change the
build most.

---

## 1. The one idea that makes this tractable: a module contract

Six enterprise modules + an agent + scraping + a 3D frontend is not a weekend of work. It is
only tractable if **every module implements the same interface.** Then:

- the **hub** can list, launch and theme any module without knowing what it does;
- the **agent** can call any module as a tool, because every module exposes tools the same way;
- the **frontend** can render any module's predictions *and* its explanations with shared
  components, because they share one shape;
- **adding module 7 costs one directory**, not a refactor.

```python
# core/contract.py  (sketch)
class Module(Protocol):
    id: str                       # "fraud" | "churn" | "rag" | "maintenance" | ...
    title: str
    summary: str
    accent: str                   # drives that body's colour across the hub

    def schema(self) -> ModuleSchema: ...        # typed inputs/outputs → auto UI + auto agent tools
    def predict(self, req: PredictRequest) -> Prediction: ...
    def explain(self, pred: Prediction) -> Explanation: ...   # optional but strongly encouraged
    def tools(self) -> list[AgentTool]: ...      # what the LangGraph agent may call
    def stream(self) -> AsyncIterator[Event]: ... # optional live feed (fraud, IoT)
    def train(self, cfg: TrainConfig) -> TrainReport: ...     # optional, runs in CI not in the API
```

The **`Explanation`** type is the quiet hero. One shape for feature importances (SHAP for
tabular, attention/gradient for nets, retrieval provenance for RAG) means a single React
component renders "why did the model say this?" for *every* module. That is the visible proof
that the bodies are connected — not just linked from a nav bar.

**Rule of the build:** nothing enters `modules/` unless it satisfies the contract. This is
what keeps the repo from becoming six unrelated Streamlit apps.

---

## 2. Topology — separate storefronts, one deploy

**The requirement, restated:** clicking a body should land you somewhere that *feels like a
different website* — the way a retailer's electronics vertical is its own storefront rather
than a tab in the same shop. Not shared chrome with a nav item.

Two ways to get that, and they cost very differently on a $0 budget:

**A. Independent *services*** — each body its own deployable on its own subdomain. True
isolation, real microservices on the CV. But 7 cold-starting free containers that sleep after
15 minutes is a fragile demo and a miserable maintenance loop.

**B. Independent *root layouts*, one deploy** — one Next.js app where every body owns a
completely separate document. **This is what shipped.**

### How B actually produces different websites

Verified against the Next.js source (`docs/.../route-groups.mdx`), not assumed:

> A route group can be created by wrapping a folder's name in parenthesis: `(folderName)`.
> Use cases: … **Defining multiple root layouts.**

So there is **no** top-level `app/layout.tsx`. Instead:

```
app/
├─ (hub)/
│  ├─ layout.tsx     ← its own <html>/<body>, dark volumetric theme, 3D graph
│  └─ page.tsx       ← /
└─ (churn)/
   ├─ layout.tsx     ← its own <html>/<body>, light warm theme, own nav, own metadata
   └─ churn/page.tsx ← /churn
```

Each root layout renders its own `<html>` and `<body>`, so the two are genuinely different
documents. Verified by diffing the served HTML:

| | `/` (hub) | `/churn` (storefront) |
|---|---|---|
| `<html>` | `lang="en" class="dark"` | `lang="en"` |
| `<body>` | `min-h-screen overflow-x-hidden bg-void` | `min-h-screen bg-[#fbfaf7] text-[#1c1917] antialiased` |
| `<title>` | Saros — AI/ML/Data-Science agent platform | Churn & Retention — Saros |
| `theme-color` | `#05060a` | `#fffbeb` |
| 3D canvas / grid floor | present | absent |

Next.js also documents the caveat that makes this *work in our favour*:

> **Full page load**: If you navigate between routes that use different root layouts, it'll
> trigger a full page reload.

That hard navigation is precisely the "you have been taken to a different site" feeling. It is
a limitation of route groups that this design turns into the intended effect.

Two further documented constraints, both respected here:
- Routes in different groups must not resolve to the same URL path.
- With multiple root layouts and no top-level layout, `/` must live inside one of the groups —
  it lives in `(hub)`.

### Subdomains later, for free

`next.config.ts` already carries the host-based rewrite, and `has: [{ type: 'host' }]` is
confirmed present in the Next.js source (`test/e2e/custom-routes/next.config.js:167`). Set
`CHURN_HOST=churn.yourdomain.dev` and that hostname renders the churn root layout — still one
deploy. **Not yet exercised against a real domain**, so treat it as wired-but-unverified.

### Why the backend stays modular rather than split

`modules/<id>/` is self-contained: `data.py`, `train.py`, `service.py`, `__init__.py` exporting
`MODULE`, plus tests. The rule that makes it real rather than nominal:

> **A module may import `core`, never another module.**

Enforced by AST scan in `tests/test_contract.py::test_module_does_not_import_another_module`,
parameterised over every package under `modules/`. A new body that reaches sideways fails CI.

Because every body already talks only through the contract, promoting one to a standalone
service later is a packaging change, not a rewrite.

## 3. The hub: agent as the connective tissue

The hub is not a landing page with six links. The hub is an **agent that can reach every
body.**

```
                       ┌──────────────────────────┐
   user ──────────────▶│   LangGraph hub graph     │
                       │  plan → route → act → cite │
                       └──────┬───────────┬────────┘
              ┌───────────────┼───────────┼───────────────┐
              ▼               ▼           ▼               ▼
         ┌────────┐     ┌─────────┐  ┌──────────┐   ┌─────────┐
         │  RAG   │     │  Fraud  │  │   Churn  │   │Scraper  │  ... 6+
         └────────┘     └─────────┘  └──────────┘   └─────────┘
              │               │           │               │
              └───────────────┴─────┬─────┴───────────────┘
                                    ▼
                        unified Prediction + Explanation
                                    ▼
                          one shared render layer
```

Each module registers its `tools()` into a central registry at boot. So a user can type
*"scrape the latest pricing from these three competitors, feed it to the pricing model, and
tell me which SKU to move"* — and the hub plans across three bodies, then cites each one.

**That** is the demo that separates this from the 12★ fragment repos.

---

## 4. Stack (all verified active 2026-10-10)

| Layer | Choice | Why |
|---|---|---|
| Orchestrator | **LangGraph** 43,019★ | Resilient stateful graphs, checkpoints, human-in-loop |
| Provider abstraction | **LangChain** 147,559★ | Swap Groq/Gemini/OpenAI/Ollama via env |
| RAG internals | **LlamaIndex** 52,458★ | Best document ingestion pipeline |
| Scraping | **Firecrawl API** 190,163★ | Free tier, zero self-hosting |
| Vectors | **pgvector on Supabase** | Persistent + free (Chroma dies on ephemeral disks) |
| Tabular ML | scikit-learn 67,522★ · XGBoost 28,847★ | Baselines + production models |
| Imbalance | imbalanced-learn 7,127★ | SMOTE for fraud/churn |
| Deep learning | PyTorch 104,083★ | LSTM for maintenance/forecasting |
| XAI | SHAP 25,803★ + Fairlearn | Audit + bias |
| Streaming | **Redis Streams** (Upstash) | Real pub/sub without Kafka's RAM bill |
| Tracking | MLflow 28,337★ | Experiments |
| Orchestration | Prefect 24,001★ | Lighter than Airflow (47,138★ but too heavy free) |
| Frontend | Next.js 143,105★ · Tailwind 97,851★ · react-three-fiber 32,843★ · **motion** 33,940★ · ECharts 67,469★ | Note: `framer/motion` is now `motiondivision/motion`; import from `motion/react` |

---

## 5. Frontend: the 3D / animation brief, made honest

You want background video, 3D, animation. Yes — with three engineering constraints that make
it *look* better rather than worse:

1. **The hub gets the spectacle; the bodies get the clarity.** A WebGL hero (particle field /
   neural-mesh in react-three-fiber) on the hub, then each module route drops to a fast,
   dense, information-rich layout. A 60 fps dashboard beats a pretty one you cannot read.
2. **3D as data, not decoration.** The strongest version: the hub renders the module graph in
   3D — six orbiting nodes wired to the core — and hovering a node *is* the navigation. The
   3D scene then earns its frame budget by being the actual interface.
3. **Budget the frame rate.** `motion` for entrances and micro-interactions, `framer`-style
   scroll reveals, and *lazy-mount* the canvas (`<Canvas>` behind an intersection observer,
   `dpr={[1, 2]}`, `frameloop="demand"` where possible). Respect
   `prefers-reduced-motion`. Background video: use a short, muted, looping, **≤3 MB**
   compressed clip or a generated gradient mesh — a 40 MB stock video on a free Vercel tier
   is the difference between "wow" and "why is this lagging".

---

## 6. Phased plan — a walking skeleton first

Do **not** build six modules in parallel. Prove the contract end-to-end with one module, then
clone the shape.

| Phase | Deliverable | Why this order |
|---|---|---|
| **0** | Monorepo, contract, hub shell, FastAPI gateway, **churn** module end-to-end | Churn has the cleanest public data and the fastest train loop. Proves contract + explain UI + deploy. |
| **1** | Agent core (LangGraph) + tool registry + **RAG** body | The connective tissue. Makes the hub an agent, not a menu. |
| **2** | **Fraud** (Redis Streams + SMOTE + live alerts) + **scraper** body | First "real-time" story + first autonomous data acquisition. |
| **3** | **Predictive maintenance** (LSTM + anomaly) + **pricing** (H3/OSRM) | Time series + geospatial. Heavier data. |
| **4** | **XAI/healthcare** body + Fairlearn bias audits | Depends on the `Explanation` type being mature. |
| **5** | 3D hub graph, background video, MLflow, Prefect, CI, public deploy | Polish + proof of ops. |

Every phase ends **deployed**. A walking skeleton online in week 1 is worth more than a
perfect local build in month 3 — and it is what you show people.

---

## 7. Decisions — settled

| # | Question | Decision |
|---|---|---|
| 1 | Agent brain | **Free API tiers** (Groq + Gemini) behind LangChain, swappable via `.env`. |
| 2 | Topology | **Separate root layouts, one deploy** — see §2. Subdomain rewrites pre-wired. |
| 3 | First body | **Churn**, end-to-end, as the template the others copy. |
| 4 | Data | **Real public dataset where it is good**, synthetic generator as the fallback. |

## 8. What shipped in phase 0

| Piece | Where | State |
|---|---|---|
| Module contract + registry | `core/contract.py` | 12 conformance rules, enforced in CI |
| Shared output types | `core/types.py` | `Prediction`, `Explanation` with an explicit `space` |
| Churn body | `modules/churn/` | real 3,333-row dataset, GBM + SMOTE, sigmoid calibration, TreeSHAP |
| Generic gateway | `apps/api/main.py` | zero churn-specific code; a new body gets an API for free |
| Hub | `apps/web/app/(hub)/` | 3D module graph as navigation, dark theme |
| Churn storefront | `apps/web/app/(churn)/` | light theme, live scorer, portfolio ranking, model card |
| Dataset fetch | `data/fetch.sh` | no Kaggle account, no credentials |
| CI + deploy | `.github/workflows/ci.yml`, `deploy/Dockerfile.api`, `render.yaml`, `apps/web/vercel.json` | |

**Held-out results on the real dataset** (from `GET /api/v1/modules/churn/metrics`, not a
notebook): PR-AUC **0.677** against a 0.145 always-positive baseline — **4.67× lift**; ROC-AUC
0.856; threshold 0.41 chosen by F1 on a reserved fold; test confusion tn 674 / fp 39 / fn 47 /
tp 74.

Four genuine bugs were found and fixed by the test suite rather than by inspection; they are
recorded in `docs/BUILD-NOTES.md` because each one is a lesson worth not relearning.
