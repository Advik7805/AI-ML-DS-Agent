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
