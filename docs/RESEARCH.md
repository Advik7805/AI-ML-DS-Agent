# GitHub Landscape Research — AI/ML/DS Agent Platform

> Compiled 2026-10-10 via the GitHub Search API. Star counts and `pushed_at` dates are
> measured, not remembered. Anything marked **MISS** was a wrong-guess slug I could not
> verify — do not cite those without re-checking.

## 1. The headline finding

**This project does not exist yet.** The landscape splits into two halves with nothing in
between:

| Half | What it looks like | Example stars |
|---|---|---|
| **Frameworks** — battle-tested, generic, huge | `langchain` 147,559★ · `langgraph` 43,019★ · `firecrawl` 190,163★ | 30k–190k★ |
| **Fragment projects** — single-purpose, abandoned-ish | `telechurnAi` 18★ · `retainiq` 12★ · `iot-predictive-maintenance` 15★ | 2–25★ |

Searches for the *combination* ("ai agent dashboard nextjs fastapi", "churn prediction
dashboard streamlit", "predictive maintenance iot dashboard") returned **only 2–22★ repos**.
Every one of them is a single module. Nobody has assembled the senior-data-scientist
toolchain into one connected platform with a real frontend.

**That is the gap this repo fills.** The moat is not the ML — the ML is commodity. The moat
is *integration quality* + *presentation* + *one coherent contract between modules*.

---

## 2. Frameworks worth building on (verified active as of 2026-10-10)

### Agent orchestration — pick ONE

| Repo | Stars | Last push | Verdict |
|---|---|---|---|
| `langchain-ai/langgraph` | 43,019 | 2026-10-10 | **RECOMMENDED.** "Build resilient agents." Stateful graphs, checkpoints, human-in-the-loop, persistence. Best fit for a hub that routes to many specialist bodies. |
| `crewAIInc/crewAI` | 59,533 | 2026-10-09 | Role-playing crews. More stars, but opinionated role/prompt abstraction — harder to bolt 6 heterogeneous ML modules onto. |
| `microsoft/autogen` | 61,342 | **2026-04-15** | **Avoid.** 6 months since last push while everything else is same-day. Momentum has left. |
| `langchain-ai/langchain` | 147,559 | 2026-10-10 | Use as the *provider abstraction* layer under LangGraph, not as the orchestrator. |
| `run-llama/llama_index` | 52,458 | 2026-10-08 | Best-in-class for the RAG module specifically (document ingestion, chunking, indices). |
| `deepset-ai/haystack` | 26,715 | — | Strong alternative to LangChain for RAG; "context-engineered, production-ready". |
| `pipecat-ai/pipecat` | 16,336 | — | Voice + realtime multimodal. Only if you add voice later. |

**Decision: LangGraph as the orchestrator, LangChain for provider abstraction, LlamaIndex
inside the RAG module.** This keeps the hub thin and lets each body own its own stack.

### Web scraping / browser agents

| Repo | Stars | Notes |
|---|---|---|
| `firecrawl/firecrawl` | 190,163 | Same-day active. The standard for "give an AI agent the web". Has a free API tier → **no self-hosting needed on a free plan.** |
| `browser-use/browser-use` | 117,538 | Agents that drive a real browser. Heavier; needs Playwright + RAM. |
| `h4ckf0r0day/obscura` | 28,747 | Headless browser built for AI agents / scraping. |
| `dzhng/deep-research` | 19,779 | Iterative deep-research agent. Great *pattern* to copy for a research tool. |
| `firecrawl/open-agent-builder` | 2,639 | Visual drag-and-drop scraping pipelines — a UX reference for our scraping body. |

**Decision: Firecrawl API for scraping (free tier, zero infra) with a `crawl4ai`/Playwright
fallback for self-hosting.** Do not build a scraping engine.

### Vector / retrieval

| Repo | Stars | Notes |
|---|---|---|
| `qdrant/qdrant` | 35,008 | High-perf vector DB. Needs a server → costs RAM on free tiers. |
| `chroma-core/chroma` | 29,472 | "Search infrastructure for AI". Embedded mode = zero infra, but **filesystem is ephemeral on free hosts.** |
| `weaviate/weaviate` | 16,877 | Multi-modal vector DB. |
| **pgvector on Supabase** | (`supabase` 111,305) | **RECOMMENDED for free deploy.** Postgres + vector index, persistent, on Supabase's free tier. Solves the ephemeral-disk problem that kills Chroma on Render/Railway. |

**Decision: pgvector via Supabase.** Persistent storage for free, one database instead of
two, and you get SQL joins between vectors and relational data — useful for the churn and
fraud modules.

### ML / MLOps

| Repo | Stars | Notes |
|---|---|---|
| `pytorch/pytorch` | 104,083 | LSTM for predictive maintenance. |
| `huggingface/transformers` | 167,115 | Embeddings + small local models. |
| `scikit-learn/scikit-learn` | 67,522 | Baselines for every module. |
| `dmlc/xgboost` | 28,847 | Churn + fraud. |
| `shap/shap` | 25,803 | XAI module. "Game theoretic approach to explain the output of any ML model." |
| `mlflow/mlflow` | 28,337 | Experiment tracking. Self-host free. |
| `PrefectHQ/prefect` | 24,001 | Pipeline orchestration. Lighter than Airflow → better for free tiers. |
| `apache/airflow` | 47,138 | Too heavy for a student free deploy. Skip. |
| `scikit-learn-contrib/imbalanced-learn` | 7,127 | **SMOTE lives here.** Last push 2026-06-29 — slower cadence but stable/mature. |
| `dagster-io/dagster` | 16,258 | Prefect alternative; asset-based. |

**Note:** `microsoft/LightGBM` and `imbalanced-learn/imbalanced-learn` returned **MISS** on
the API — my slugs were wrong. imbalanced-learn's real home is under
`scikit-learn-contrib` (verified above). Re-verify LightGBM's slug before adding it.

### Streaming (fraud / real-time)

| Repo | Stars | Notes |
|---|---|---|
| `apache/kafka` | 33,924 | The reference. **Too heavy for a free tier.** |
| `risingwavelabs/risingwave` | 9,365 | "Event streaming platform for agentic AI." SQL-native streaming — a genuinely good modern fit. |
| `redis/redis` | 76,699 | **RECOMMENDED for free deploy.** Redis Streams gives you a real pub/sub pipeline; Upstash's free tier is serverless. |
| `celery/celery` | 28,939 | Task queue for async jobs (training runs, scrapes). |

**Decision: Redis Streams (Upstash free tier) for the "real-time" story, not Kafka.** You get
genuine streaming semantics — a live transaction feed, sliding-window features, live anomaly
alerts — without needing a broker that eats your entire RAM budget. Be honest in the README
that this is a streaming *pattern*, not a Kafka lakehouse.

### Frontend

| Repo | Stars | Notes |
|---|---|---|
| `vercel/next.js` | 143,105 | App Router. Deploys free on Vercel. |
| `tailwindlabs/tailwindcss` | 97,851 | Design system backbone. |
| `pmndrs/react-three-fiber` | 32,843 | **3D.** React renderer for Three.js. Same-day active. |
| `motiondivision/motion` | 33,940 | **Animations.** This is the current home of Framer Motion — `framer/motion` returned **MISS**. Import from `motion/react`. |
| `apache/echarts` | 67,469 | Heavy-duty charts (candlesticks, graphs, gauges, heatmaps). |
| `plotly/plotly.py` | 18,828 | Python-side plotting; pairs with SHAP. |

**Decision: Next.js + Tailwind + react-three-fiber (+ `@react-three/drei`) + motion + ECharts.**
Verified current package name for animations is `motion`, not `framer-motion`.

---

## 3. Fragment repos worth *reading* (not depending on)

These are small, but they show what a single-module implementation looks like. Useful as
checklists for features to match and beat:

- `ashishallu/retainiq` (12★) — churn + GBM + SMOTE + CLV. Directly comparable to module 6.
- `chauhan-varun/telechurnAi` (18★) — telecom churn.
- `dhamini-hue/Real-Time-Industrial-IoT-Predictive-Maintenance-System` (22★) — module 3.
- `karimosman89/iot-predictive-maintenance` (15★) — module 3.
- `MuhammadZaidSaqib/SentinelX-Real-Time-Fraud-Intelligence` (11★) — module 2.
- `Victor-Kipruto-Rop/mpesa_safaricom-fraud_anomaly_detection` (34★) — Kafka/Flink fraud engine.
- `oegedijk/explainerdashboard` (2,511★) — the *only* well-built XAI dashboard found. Study its
  SHAP presentation; ours should match or exceed it inside a nicer shell.
- `Stream-AD/MIDAS` (776★) / `MStream` (118★) — streaming anomaly detection algorithms.
- `JulesBelveze/time-series-autoencoder` (714★) — LSTM-AE for forecasting.
- `AIStream-Peelout/flow-forecast` (2,304★) — PyTorch time-series lib.
- `AmirhosseinHonardoust/AI-Assistant-Satisfaction-Prediction-Engine` (18★) — SMOTE + behavioral features.

**The bar is low.** A clean, connected platform with 6 modules beats every one of these on
presentation alone.

---

## 4. Free deployment map (student budget = $0)

| Component | Free option | Hard limits to design around |
|---|---|---|
| Frontend | **Vercel** (Hobby) | Generous. Next.js native. |
| Backend API | **Render** free | Sleeps after 15 min idle → first request is slow. **Ephemeral filesystem.** 512 MB RAM. |
| ML-heavy demo | **Hugging Face Spaces** (free CPU) | Better RAM than Render for model inference. Docker SDK supported. Sleeps when idle. |
| Database + vectors | **Supabase** free | 500 MB DB, 1 GB file storage. Persistent. **This is why we chose pgvector.** |
| Cache / streams / queue | **Upstash Redis** free | Serverless, per-request pricing, generous free tier. |
| LLM inference | **Groq** / **Gemini** free tiers | Rate-limited but real. Or **Ollama** locally for dev. |
| Object storage | **Cloudflare R2** free egress | For scraped artifacts / model files. |
| CI | **GitHub Actions** | 2,000 min/month free for private, unlimited public. |

**Two consequences of the 512 MB / ephemeral-disk reality:**
1. **No heavy model files in the API container.** Train in a notebook/GitHub Action, ship a
   small serialized model (<20 MB), or run inference on HF Spaces.
2. **No local vector DB.** Everything persistent goes to Supabase.

---

## 5. Open questions this research could not answer

- Whether you have (or will pay for) an LLM API key, or want it to run free/local only.
- Whether "each component has its own site" means *separately deployable services* or
  *one deploy with independent modules* — very different infra cost.
- Which of the six modules matters most to you first.

These are in `docs/ARCHITECTURE.md` as explicit decision points.
