# Deploying Nexus for $0

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
   path, `/health` as the health check, and all 18 env vars.
3. Render will prompt for the eight secrets marked `sync: false`. **Leave them blank.** Every one
   has a labelled mock fallback.
4. Deploy. You get something like `https://nexus-api.onrender.com`.

Verify it:

```bash
curl https://nexus-api.onrender.com/health
# {"status":"ok","modules_loaded":[...7 ids...],"load_errors":{}}

curl https://nexus-api.onrender.com/api/v1/integrations
```

**What to expect on the free tier:** the service sleeps after 15 minutes idle and the next
request takes ~60 seconds. That is not a bug and it is not fixable without paying — in 2026
Render is the only one of Render/Railway/Fly.io with an ongoing free tier.

Two things already handle it:

- **The frontend degrades.** Every page catches the fetch failure and renders a notice rather
  than a blank screen.
- **Keep-alive keeps it warm.** Add `NEXUS_API_URL` as a *repository secret* in GitHub
  (Settings → Secrets and variables → Actions) set to `https://nexus-api.onrender.com`. The
  [`keep-alive`](../.github/workflows/keep-alive.yml) workflow then pings `/health` every 10
  minutes, inside the 15-minute window. Without the secret it skips cleanly rather than failing.

**Do not** put anything durable on Render's free Postgres — it expires after 30 days.

---

## 3. Frontend → Vercel

1. Vercel → **Add New Project** → import the repo. Root directory: `apps/web`.
2. **Set `API_ORIGIN` before the first build.** This is the step people miss:

   ```
   API_ORIGIN = https://nexus-api.onrender.com
   ```

   No trailing slash. Without it the build **fails on purpose**, with a message naming the
   variable. That is deliberate: with `API_ORIGIN` unset, every `/api` request proxies to
   `http://127.0.0.1:8000`, which on Vercel is the serverless function's own loopback. The hub
   would render, every storefront would render, and every API call would fail with no clue why.
   A site that looks deployed and answers nothing is worse than a build error.

3. Deploy.

The browser never learns the API's address — `next.config.ts` rewrites `/api/*` and `/health`
server-side, which is also what makes the preview URLs work and removes CORS from the picture.

---

## 4. Light up the integrations, one at a time

Nothing below is required. Add a key, set its flag, redeploy. The hub's **External services**
panel and `GET /api/v1/integrations` show what is live and what is still mocked.

| Service | Env vars | Free tier | What it unlocks |
|---|---|---|---|
| **Groq** | `GROQ_API_KEY`, `NEXUS_LLM=groq` | free tier, per-minute limits | real reasoning in the hub agent |
| **Gemini** | `GEMINI_API_KEY`, `NEXUS_LLM=gemini` | free tier, per-day limits | same, second vendor |
| **Tavily** | `TAVILY_API_KEY`, `NEXUS_TAVILY_LIVE=1` | **1,000 credits/month, students free** | `web_search`, `extract_url` |
| **Firecrawl** | `FIRECRAWL_API_KEY`, `NEXUS_FIRECRAWL_LIVE=1` | 500 credits **one-time** | `render_url` (JS rendering) |
| **Upstash** | `UPSTASH_REDIS_REST_URL`, `UPSTASH_REDIS_REST_TOKEN`, `NEXUS_UPSTASH_LIVE=1` | 10k commands/day | shared rate limiting |
| **Supabase** | `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`, `NEXUS_SUPABASE_LIVE=1` | 500 MB | extra knowledge-base documents, no redeploy |

Suggested order, best value first:

1. **Groq or Gemini.** Biggest visible change — the agent stops templating answers from tool
   output and starts actually reasoning. A missing key falls back to the mock provider rather
   than throwing, so this is safe to experiment with.
2. **Tavily.** 1,000 credits a month with no card is enough for real development. Basic search
   is 1 credit; keep `NEXUS_TAVILY_SEARCH_DEPTH=basic` — `advanced` doubles it.
3. **Upstash.** Worth doing *before* you share the link publicly. Without it the rate limit is
   per-process, so one crawler can exhaust the API's request budget.
4. **Firecrawl.** Only if a page genuinely needs a browser. Tavily's extract covers most cases
   on a friendlier allowance, and Firecrawl's 500 credits never refill.
5. **Supabase.** Only if you want to add knowledge-base documents without redeploying. It is a
   plain-text document table, *not* a vector store — at 25 passages TF-IDF already retrieves
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
| First request after idle takes ~60s | Render free tier spin-down | Normal. Add the `NEXUS_API_URL` secret |
| Agent answers look templated | `NEXUS_LLM=mock` | Set a key and `NEXUS_LLM=groq`/`gemini` |
| Web search returns placeholders | Tavily key set but flag off | Also set `NEXUS_TAVILY_LIVE=1` |
| `429` responses | Rate limit | Expected; `X-RateLimit-*` headers say which backend counted |
| A body is missing from the hub | It failed to import | `GET /health` → check `load_errors` |

The last one is why CI asserts the exact body list. It is the only failure on this page that the
UI cannot show you.
