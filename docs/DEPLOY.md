# Deploying Saros for $0

Every step here is verified against the code in this repo. Nothing requires a credit card.

**Order matters.** Deploy the API first and get its URL, because the frontend build fails
without it — deliberately, see [step 3](#3-frontend--vercel).

---

## 1. Push the repo

The CI workflow runs on every push to `main` or an `arena/**` branch.

```bash
git push origin main
```

Watch it under **Actions**. It must be green before you deploy — it asserts that all seven
bodies are discovered and that all 20 tool names are unique, which is the failure mode that is
otherwise invisible (a body that fails to import lands in `load_errors` and the gateway still
serves the other six).

---

## 2. API → Render free

1. Render dashboard → **New +** → **Blueprint** → point it at this repo.
2. It reads [`render.yaml`](../render.yaml), which already declares the service, the Dockerfile
   path, `/health` as the health check, and all 26 env vars.
3. Render will prompt for the 12 secrets marked `sync: false`. **Leave them blank.** Every one
   has a labelled mock fallback.
4. Deploy. You get something like `https://saros-api.onrender.com`.

Verify it:

```bash
curl https://saros-api.onrender.com/health
# {"status":"ok","modules_loaded":[...7 ids...],"load_errors":{}}

curl https://saros-api.onrender.com/api/v1/integrations
```

**What to expect on the free tier:** the service sleeps after 15 minutes idle and the next
request takes ~60 seconds. That is not a bug and it is not fixable without paying — in 2026
Render is the only one of Render/Railway/Fly.io with an ongoing free tier.

Two things already handle it:

- **The frontend degrades.** Every page catches the fetch failure and renders a notice rather
  than a blank screen.
- **Keep-alive keeps it warm.** Add `SAROS_API_URL` as a *repository secret* in GitHub
  (Settings → Secrets and variables → Actions) set to `https://saros-api.onrender.com`. The
  [`keep-alive`](../.github/workflows/keep-alive.yml) workflow then pings `/health` every 10
  minutes, inside the 15-minute window. Without the secret it skips cleanly rather than failing.

**Do not** put anything durable on Render's free Postgres — it expires after 30 days.

---

## 3. Frontend → Vercel

1. Vercel → **Add New Project** → import the repo. Root directory: `apps/web`.
2. **Set `API_ORIGIN` before the first build.** This is the step people miss:

   ```
   API_ORIGIN = https://saros-api.onrender.com
   ```

   No trailing slash. Without it the build **fails on purpose**, with a message naming the
   variable. That is deliberate: with `API_ORIGIN` unset, every `/api` request proxies to
   `http://127.0.0.1:8000`, which on Vercel is the serverless function's own loopback. The hub
   would render, every storefront would render, and every API call would fail with no clue why.
   A site that looks deployed and answers nothing is worse than a build error.

3. Deploy.

The browser never learns the API's address — `next.config.ts` rewrites `/api/*` and `/health`
server-side, which is also what makes the preview URLs work and removes CORS from the picture.

### Verify it in a browser, not with curl

A deploy can return a perfect 200 for every route and still be unusable: if the page never
hydrates, every control is inert and the HTML looks identical. That is a real failure mode here —
see note 8 in [BUILD-NOTES.md](BUILD-NOTES.md) — so there is a script for it:

```bash
cd apps/web
npm install --no-save puppeteer-core @sparticuz/chromium
BASE=https://saros-hub.vercel.app node scripts/browser-check.mjs   # the live app; use your own URL for a fork
```

It loads all eight routes, asserts each one hydrates and makes no failing API call, clicks through
all seven body chips in the upload studio, and drops a file if you give it one
(`FILE=/path/to.csv`). Non-zero exit means something a screenshot would not show you.

### If you serve a preview URL from `next dev`

Add the host to `allowedDevOrigins` — it is already covered for `localhost`, `127.0.0.1`, `0.0.0.0`,
`*.e2b.app` and `*.vercel.app`. For anything else, set `SAROS_DEV_ORIGINS=host-a,host-b` in the
environment before starting the dev server. Next 16 blocks dev resources from unknown origins, and
because the dev client waits for its HMR socket before hydrating, a blocked request means the page
renders and no control on it works. Production builds are unaffected.

---

## 4. Light up the integrations, one at a time

Nothing below is required. Add a key, set its flag, redeploy. The hub's **External services**
panel and `GET /api/v1/integrations` show what is live and what is still mocked.

| Service | Env vars | Free tier | What it unlocks |
|---|---|---|---|
| **Groq** | `GROQ_API_KEY`, `SAROS_LLM=groq` | free tier, per-minute limits | real reasoning in the hub agent |
| **Gemini** | `GEMINI_API_KEY`, `SAROS_LLM=gemini` | free tier, per-day limits | same, second vendor |
| **Tavily** | `TAVILY_API_KEY`, `SAROS_TAVILY_LIVE=1` | **1,000 credits/month, students free** | `web_search`, `extract_url` |
| **Jina Reader** | `JINA_API_KEY` *(optional)* | **~20 req/min with no key**; 500 RPM + 10M tokens with a free key | `render_url` — works with zero configuration |
| **Firecrawl** | `FIRECRAWL_API_KEY`, `SAROS_FIRECRAWL_LIVE=1` | 500 credits **one-time** | `render_url` fallback, only if configured |
| **MLflow** | `MLFLOW_TRACKING_URI` | self-hosted, free | experiment history across redeploys |
| **Upstash** | `UPSTASH_REDIS_REST_URL`, `UPSTASH_REDIS_REST_TOKEN`, `SAROS_UPSTASH_LIVE=1` | 10k commands/day | shared rate limiting |
| **Supabase** | `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`, `SAROS_SUPABASE_LIVE=1` | 500 MB | extra knowledge-base documents, no redeploy |

**Note that JS rendering needs nothing at all.** `render_url` goes through Jina Reader's
anonymous endpoint by default, so scraping works on a fresh deploy with no key and no signup. A
free Jina key only raises the limit from ~20 to 500 requests/minute.

Suggested order, best value first:

1. **Groq or Gemini.** Biggest visible change — the agent stops templating answers from tool
   output and starts actually reasoning. A missing key falls back to the mock provider rather
   than throwing, so this is safe to experiment with.
2. **Tavily.** 1,000 credits a month with no card is enough for real development. Basic search
   is 1 credit; keep `SAROS_TAVILY_SEARCH_DEPTH=basic` — `advanced` doubles it.
3. **Upstash.** Worth doing *before* you share the link publicly. Without it the rate limit is
   per-process, so one crawler can exhaust the API's request budget.
4. **Firecrawl.** Almost certainly skip it. Jina already renders JavaScript for free, and
   Firecrawl's 500 credits never refill. It only earns its place if you need its crawl, sitemap
   or structured-extract endpoints.
5. **Supabase.** Only if you want to add knowledge-base documents without redeploying. First run
   [`deploy/supabase_schema.sql`](../deploy/supabase_schema.sql) in the Supabase SQL editor — it
   is idempotent and seeds one row you can verify against
   `GET /api/v1/agent/tools/list_knowledge_sources` (which reports `external_live`). Then set the
   three env vars above. It is a plain-text document table, *not* a vector store — at 25 passages TF-IDF already retrieves
   correctly and pgvector would add a round trip and an embedding dependency for nothing. Create
   the table with the schema at the top of `core/integrations/supabase.py`; the in-repo six
   documents always load regardless, and a database outage degrades to them rather than to an
   empty knowledge base.

**Optional storefront backdrop video.** Set `BACKDROP_VIDEO_URL` to a hosted file and every
storefront layers it beneath the 3D field, tinted to that body's accent. It is unset by default
on purpose: a background video is 5–30 MB, and committing one means every clone and every deploy
carries it. Use a CDN — Cloudflare R2's free tier is 10 GB storage and 10 GB egress, which covers
it. Only `http(s)` URLs are accepted; a `javascript:` URI or a filesystem path is ignored and the
procedural field renders alone. The variable is read server-side, so it never enters the client
bundle.

Full list with commentary: [`.env.example`](../.env.example).

---

## Two rules the platform enforces

Worth understanding before you debug a "why is this still mock?" question.

1. **A key alone never enables a metered service.** Each also needs its `*_LIVE=1` flag. So a key
   pasted in for a local experiment cannot make a demo billable. If you set `TAVILY_API_KEY` and
   still see mock results, that is why.

2. **Nothing is silently faked.** A mock labels itself in the API payload (`backend: "mock"`) and
   in the UI. An agent that cannot tell simulated retrieval from real retrieval will confidently
   cite things that never happened — so a placeholder says so, loudly.

---

## If something is wrong

| Symptom | Cause | Fix |
|---|---|---|
| Pages render, every API call fails | `API_ORIGIN` unset or wrong | Set it in Vercel; no trailing slash |
| Pages render, **nothing is clickable** | Dev resources blocked for this origin (dev server only) | Add the host to `allowedDevOrigins` / set `SAROS_DEV_ORIGINS`, restart `next dev` |
| A body chip or file drop does nothing | The page never hydrated — same cause as above | Check the browser console for a blocked `/_next/hmr` request; run `scripts/browser-check.mjs` |
| First request after idle takes ~60s | Render free tier spin-down | Normal. Add the `SAROS_API_URL` secret |
| Agent answers look templated | `SAROS_LLM=mock` | Set a key and `SAROS_LLM=groq`/`gemini` |
| Web search returns placeholders | Tavily key set but flag off | Also set `SAROS_TAVILY_LIVE=1` |
| `429` responses | Rate limit | Expected; `X-RateLimit-*` headers say which backend counted |
| A body is missing from the hub | It failed to import | `GET /health` → check `load_errors` |

The last one is why CI asserts the exact body list. It is the only failure on this page that the
UI cannot show you.
