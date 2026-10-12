# What you can do here

Generated from the running app (API + browser checks). Every item was executed, not just built.

## 1. Ask the hub agent
- **Plain English, no setup:** "Rank the top 5 customers most likely to churn", "How good is the fraud
  model, and at what operating point?", "Which engines need maintenance first?", "What is the refund
  window, and which document says so?"
- The agent plans, calls tools, and shows a trace of which tools ran. Without a key it uses a
  deterministic mock that still returns **real model outputs** and says so.
- With `GROQ_API_KEY` or `GEMINI_API_KEY` set (yours are, so this is live) it reasons in prose.

## 2. Bring your own data (hub → "Your data")
Pick any of the 7 bodies, drop a file (≤10 MB, in memory, never written to disk):
- **Churn / Fraud / Diagnostics:** CSV of your rows → real PR-AUC / ROC-AUC / precision / recall when your
  file has a label column, a distribution report (and **no** accuracy claim) when it does not.
  Column mapping shows exactly which of your columns the model used and which it ignored.
- **Maintenance:** the real 26-column CMAPSS FD001 layout, or a header CSV with the same names →
  remaining-useful-life over your telemetry.
- **Pricing:** your own price/demand log → elasticity fitted **per SKU on your data**.
- **RAG:** any `.md` / `.txt` / `.pdf` → chunked heading-wise into this session's overlay.
- **Scraper:** CSV of up to 5 URLs → fetched (mock unless a backend is configured), per row labelled live/mock.
- Then **ask about the file in the same conversation** — the upload studio's "ask" box, or the hub chat
  above it. Both carry the session id, so the answer quotes your file, not the shipped corpus.

## 3. Run any single body's console
Hub → any body → **Tool console**: pick a tool, the arguments are prefilled with real examples, press
Run. All 22 tools answer (this is now a test).

## 4. Watch decisions, not just scores
- Every body page prints **what data the decision came from** (company, dataset, origin, rows, join key).
- Churn portfolio: the target list with `NW-C-######` ids you can join to the fraud ledger.
- Fraud: stream summary + drift check (PSI per feature).
- Maintenance: fleet ranked by urgency with `NW-E-##` asset ids and per-band error.
- Pricing: per-SKU opportunity, clipped to observed prices, with confidence intervals.
- Diagnostics: SHAP drivers + a Fairlearn age-band fairness audit, flagged when the gap is bad.
- RAG: answers carry their passages and a grounding verdict; zero hits are reported as zero hits.

## 5. Operate it
- `GET /health` — instant, includes warm-up progress.
- `GET /api/v1/warmup` — per-body warm-up result and headline metric.
- `GET /api/v1/tenant`, `/api/v1/modules`, `/api/v1/integrations`, `/api/v1/runs`.
- `python -m flows.train_all` retrains everything and logs the runs.

## 6. Verify a deploy in a browser (curl cannot see this)
```bash
cd apps/web
npm install --no-save puppeteer-core @sparticuz/chromium
BASE=https://your-app node scripts/browser-check.mjs
FILE=/path/to/your.csv BASE=https://your-app node scripts/browser-check.mjs
```
Non-zero exit means something a screenshot would not show: a route that never hydrated, or a failing API call.
