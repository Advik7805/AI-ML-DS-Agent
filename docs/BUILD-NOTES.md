# Build notes — what the tests caught

Four real defects surfaced during phase 0. None were found by reading the code; all were found
by running it. They are written up because each one is the kind of mistake that produces a
model which looks excellent in a notebook and quietly fails in production.

---

## 1. Threshold selection leaked through the calibrator

**Symptom:** validation F1 `0.955` against test F1 `0.802`. A gap that wide is not
generalisation error, it is information crossing a boundary it should not.

**Cause:**
```python
X_tr, X_val, ... = train_test_split(X_train, ...)        # val comes OUT of train
model = CalibratedClassifierCV(base, ...).fit(X_train, y_train)   # ...and train is what we fit on
threshold = argmax_f1(model, X_val)                      # so the model had already seen val
```

The operating threshold was chosen on data the shipped model had trained on. The threshold is
what turns a probability into a decision, so this corrupted every downstream number.

**Fix:** four disjoint folds — fit / calibrate / threshold / test. Result: threshold-fold F1
0.713 vs test F1 0.633, a plausible gap. `test_no_leakage_between_threshold_and_report_folds`
now fails the build if the gap exceeds 0.15.

**Cost:** PR-AUC fell from 0.810 to 0.677, because the fit fold shrank from 1,874 to 1,499
rows. That drop is not a regression — the first number was fiction.

---

## 2. Isotonic calibration flattened the entire low-risk range

**Symptom:** every customer with 0–3 service calls scored **exactly 0.0208**.

**Cause:** isotonic regression fits a step function. On a calibration fold this small it
collapsed a whole region of raw scores onto a single constant. The UI then displayed a number
with two significant figures that carried zero ranking information — the worst combination,
because it *looks* precise.

**Fix:** Platt scaling (`method="sigmoid"`), which is smooth and monotone. Guarded by
`test_scores_are_spread_across_the_real_distribution`, which requires >200 distinct scores and
four distinct quantiles across the real 3,333 customers.

**Note on the test itself:** the first version of this test probed a synthetic ladder with
every other feature pinned and still failed — because gradient boosting *is* piecewise
constant, so a fixed-feature ladder legitimately repeats values (measured margins:
`−1.1856 ×3, −0.6997, +0.5066 ×3`). The right probe is the real distribution, not a ladder.

---

## 3. `cv=3` made the shipped model a different model than the one SHAP explained

**Symptom:** voice mail looked protective on the raw margin (−1.43 → −1.96) but *riskier* after
calibration (0.0295 → 0.0135 inverted to 0.0772 → 0.0906 on the affected record). A feature
cannot genuinely flip sign by passing through a monotone map.

**Cause:** `CalibratedClassifierCV(cv=3)` **clones and refits** the estimator on its own folds.
The object producing `score` was therefore not the object the `TreeExplainer` wrapped — so the
explanations were describing a sibling model. The two disagreed in low-confidence regions,
producing real inversions.

**Fix:** wrap the fitted estimator in `FrozenEstimator`, so the calibration folds can only
learn the sigmoid map and can never retrain the trees. Now `score` is a strictly monotone
function of the exact margin SHAP decomposes — asserted over 300 real customers with zero
tolerance for violations in `test_calibrated_score_is_monotone_in_the_explained_margin`.

**A wrinkle:** `cv="prefit"` — the historical way to express this — **was removed in sklearn
1.6**. `FrozenEstimator` is the replacement, and unlike a naive `clone()` it survives cloning
with fitted state intact. `requirements.txt` therefore pins `scikit-learn>=1.6`.

---

## 4. SHAP values are in log-odds, not probability

**Symptom:** the additive property did not hold against `predict_proba`.

**Cause:** `TreeExplainer` is *exact* in the model's raw margin (log-odds for classification)
and only approximate in probability space. Measured directly:

```
base_value + sum(attributions) = -0.08599333916796154
raw_model.decision_function(X) = -0.08599333916796109     ← agree to 4.5e-16
raw_model.predict_proba(X)     =  0.47851490350650244     ← does not
```

The type had documented `attribution` as "signed contribution in probability space", which was
simply false, and the UI was about to render those bars next to a percentage and imply they
summed to it.

**Fix:** `Explanation` now carries an explicit `space` field (`log_odds | probability | logits |
unspecified`), the churn body sets `space="log_odds"`, and the explanation ships a caveat saying
the bars do **not** sum to the displayed probability. `test_explanation_declares_its_attribution_space`
asserts both the label and that the two quantities genuinely differ — so the caveat cannot
become decorative.

---

## 5. A confounded comparison in a test

Not a model bug, a *test* bug, but worth recording: the voice-mail test moved
`voice_mail_plan` **and** `number_vmail_messages` together, so the message count's own effect
swamped the plan flag and made a genuinely protective feature look harmful. Changing only
`voice_mail_plan` gives margin −1.186 → −1.821 and score 0.077 → 0.035, matching the raw data
(16.7% churn without voice mail vs 8.7% with).

The lesson generalises: assert marginal effects one variable at a time, and back them with a
population-level check (`test_population_level_scores_track_the_real_churn_rates` compares
predicted vs actual group rates on all 3,333 customers).

---

## Also fixed

- **Registry discovery was not idempotent.** `uvicorn --reload` and a second test module each
  re-ran discovery, turning `/health` into a wall of `Duplicate module id 'churn'` errors.
  Re-discovering the same singleton is now a no-op.
- **`@app.on_event("startup")` is deprecated** in FastAPI; replaced with a `lifespan` handler,
  and `pyproject.toml` promotes `DeprecationWarning` to an error for first-party code so this
  cannot creep back.
- **Dead code removed:** an unused `_numeric_baseline` helper and unused imports.


---

## 6. The deploy image trained on synthetic data while claiming the real thing

**Symptom:** nothing observable in CI or locally — the defect only exists inside the Docker
build. `deploy/Dockerfile.api` runs `RUN bash data/fetch.sh || echo "dataset fetch skipped…"`,
and its comment said "Real dataset, baked in."

**Cause:** `data/fetch.sh` used `curl`, but `python:3.11-slim` ships `wget` and **not** `curl`
(verified against the upstream image's apt list). Inside the image the script dies on the first
line, the `||` swallows it, and the image silently trains on the synthetic fallbacks. Churn
PR-AUC is 0.677 on the real telco data and 0.495 synthetic — a production-only regression with a
log line that says nothing went wrong.

**Fix:** the fetch now uses `python3`'s `urllib` — the one binary a Python image guarantees.
Planting a 75-byte LFS pointer is still detected and refetched.

## 7. The hub showed working products as "planned"

**Symptom:** a user reported "everything except churn is not working." The backends were all
live; the hub was lying about them.

**Cause:** the hub page hardcoded churn as the only live tile and rendered the other six from a
static roadmap list marked *planned* with no link. The 3D graph was worse: any live node's click
ran `router.push('/churn')`, `xai_health` never matched (the list said `xai-health`), and the
scraper node did not exist.

**Fix:** tiles and graph nodes now render from the live manifests (`GET /api/v1/modules` already
returns status, accent and each body's UI route); the static list survives only as a
cold-backend fallback. `test_frontend_body_mirror_matches_the_registry` fails the build if the
mirror's ids or hrefs ever drift from the registry again.

## 8. The whole UI was decorative, and nothing said so

**Symptom:** "only able to select churn and retention and not able to add any file" — and in
general: pages that look completely finished, where clicking a body chip, a sample question or
the file drop-zone does nothing at all. No error, no toast, no console message. `curl` returned a
complete 200 with the right HTML; every API route answered; the Python suite was green.

**Cause:** `next dev`, served behind a preview host. Next 16 refuses to hand its *dev* resources
to an origin it does not recognise, so `/_next/hmr` was answered with
`Blocked cross-origin request … To allow this host in development, add it to "allowedDevOrigins"`.
The dev client waits on that socket before it hydrates, so React never attached to the DOM: the
server-rendered HTML stayed on screen, fully styled, entirely inert. The server logged the
blocked request while the browser logged a WebSocket handshake failure — neither of which
appears in a screenshot, a `curl`, or a test suite.

**Fix:** `allowedDevOrigins` in `apps/web/next.config.ts` covers localhost, `127.0.0.1`, `0.0.0.0`,
`*.e2b.app` (the sandbox preview host), `*.vercel.app`, plus anything in `SAROS_DEV_ORIGINS`.
Nine source-level assertions in `tests/test_frontend_contract.py` keep it there, because this is
the one bug that makes every feature look broken at once.

**How it was actually found:** by driving a real headless Chromium against the page and asking
whether the DOM had any React fiber attached, then clicking every visible control and checking
for a text change *or* an API call. `apps/web/scripts/browser-check.mjs` does exactly that and
now runs as a pre-deploy gate.

## 9. `GET /api/v1/agent` answered 500 with no explanation

**Symptom:** the chat header's provider badge silently never resolved.

**Cause:** the handler read `os.getenv("SAROS_LLM", "auto")` in a module that never imported `os`.
A `NameError`, so a 500 — and the frontend's `.catch()` hid it.

**Fix:** the import, plus `tests/test_routes.py`, which walks the app's own OpenAPI schema and
calls every GET route, failing on any 5xx and on any 4xx that carries an empty `detail`. A
missing import is not a subtle bug; it was subtle only because nothing called the route.

## 10. A reloaded model card reported zeros as fact

**Symptom:** `GET /api/v1/modules/maintenance/metrics` reported `n_samples: 0, positive_rate: 0.0`
after a restart, while the model was fine.

**Cause:** the load path rebuilt `MetricsReport` from scratch, and there was nothing to put in it —
the artifact never stored its evaluation. Retraining was the only way to answer, so the fallback
was to invent zeros and render them next to the model's real metrics.

**Fix:** trainable bundles now carry the report they were judged by (`core/artifacts.SCHEMA` 2 → 3,
so older pickles retrain instead of half-loading). Measured: `get_maintenance_model_metrics` went
from `n_samples: 0` to `24,178 / 12.8% positive` in 2 ms, and churn's metric call stopped
retraining the model on first use (2.3 s saved). Guarded by `tests/test_model_card.py`.

## 11. Six tool-console defects found by calling every tool

A sweep of all 22 tools over HTTP, with the exact arguments the UI prefills, produced six
"the button does nothing" bugs — all now fixed and covered by `tests/test_tools_callable.py`,
which mirrors the console's own default-argument logic:

- A renamed or unknown argument returned `500 {"detail": ""}` (an escaped `TypeError`). Tool
  dispatch now validates against the tool's own JSON Schema and answers with the accepted names.
- `scrape_url`, `web_search`, `extract_url`, `render_url`, `search_knowledge` and `optimise_price`
  answered 422 to a *prefilled* console, because an empty string is not a URL or a SKU. Tool
  schemas now carry `examples`, which is what the console prefers.
- `extract_url` iterated a single URL string character by character.
- `assess_diabetes_risk` reported pandas' "could not convert string to float: 'sample'" with no
  field named.
- The RAG console's tool `<select>` could render empty with no explanation; options were also
  unreadable on the dark surface.
- The tool console called `/explain` on bodies that ship no explainer (pricing, scraper), logging
  422s for a request that could never succeed.
