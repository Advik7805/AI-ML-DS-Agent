"""Churn body — data, model quality, and explanation integrity.

Training once per session: ~7s on 3,333 rows, so a module-scoped fixture keeps the suite fast
without mocking out the thing under test.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.types import Explanation, Prediction
from modules.churn.data import FEATURES, RAW_CHURN_CSV, TARGET, load_telco, synthesize
from modules.churn.train import ChurnModel, train_churn


@pytest.fixture(scope="module")
def trained() -> tuple[ChurnModel, object]:
    return train_churn()


@pytest.fixture(scope="module")
def module(trained):
    """A ChurnModule pre-seeded with the session's trained bundle (skips artifact I/O)."""
    from modules.churn.service import ChurnModule

    bundle, report = trained
    m = ChurnModule()
    m._bundle = bundle
    m._report = report
    return m


# --- data --------------------------------------------------------------------


def test_real_dataset_loads_and_is_real() -> None:
    assert RAW_CHURN_CSV.exists(), "data/raw/telco_churn.csv missing — run data/fetch.sh"
    df, source = load_telco()
    assert source == "real"
    assert len(df) == 3333, f"expected the canonical 3,333 rows, got {len(df)}"
    assert set(FEATURES).issubset(df.columns)
    assert TARGET in df.columns


def test_target_is_binary_and_imbalanced() -> None:
    df, _ = load_telco()
    assert set(df[TARGET].unique()) <= {0, 1}
    rate = df[TARGET].mean()
    assert 0.10 < rate < 0.20, f"expected ~14.5% churn, got {rate:.3f}"


def test_no_nans_in_features() -> None:
    df, _ = load_telco()
    assert df[list(FEATURES)].notna().all().all()


def test_charge_columns_are_collinear_so_dropping_them_is_justified() -> None:
    """The pipeline drops the four ``*_charge`` columns. Verify that claim is actually true."""
    df, _ = load_telco()
    for minutes, charge, rate in [
        ("total_day_minutes", "total_day_charge", 0.17),
        ("total_eve_minutes", "total_eve_charge", 0.085),
        ("total_night_minutes", "total_night_charge", 0.045),
        ("total_intl_minutes", "total_intl_charge", 0.27),
    ]:
        derived = (df[minutes] * rate).round(2)
        corr = np.corrcoef(df[minutes], df[charge])[0, 1]
        assert corr > 0.999, f"{charge} vs {minutes} corr={corr:.5f}, not collinear"
        assert (derived - df[charge]).abs().max() < 0.02, f"{charge} is not minutes*{rate}"
        assert charge not in FEATURES, f"{charge} should be excluded from FEATURES"


def test_synthetic_fallback_matches_the_real_schema() -> None:
    df, _ = load_telco()
    synth = synthesize(n=500, seed=7)
    assert set(FEATURES).issubset(synth.columns), "synthetic must cover every model feature"
    assert TARGET in synth.columns
    # Same shape, and a comparable positive rate — otherwise the fallback trains a different problem.
    assert 0.05 < synth[TARGET].mean() < 0.30
    assert synth[list(FEATURES)].notna().all().all()


def test_load_telco_falls_back_when_file_absent(tmp_path) -> None:
    df, source = load_telco(tmp_path / "nope.csv")
    assert source == "synthetic"
    assert len(df) > 0


# --- model quality -----------------------------------------------------------


def test_training_uses_real_data(trained) -> None:
    bundle, _ = trained
    assert bundle.source == "real"
    assert bundle.feature_names == list(FEATURES)


def test_model_beats_the_trivial_baseline(trained) -> None:
    """PR-AUC must clear the always-positive baseline by a wide margin, or it is not a model."""
    _, report = trained
    pr_auc = report.scores["pr_auc"]
    base = report.scores["baseline_pr_auc"]
    assert pr_auc > 3 * base, f"PR-AUC {pr_auc:.3f} is not 3x the {base:.3f} baseline"
    assert report.scores["roc_auc"] > 0.80


def test_threshold_is_selected_not_defaulted(trained) -> None:
    """0.50 is the wrong operating point for a 14.5% positive rate. Prove we did not default."""
    bundle, report = trained
    assert 0.05 <= bundle.threshold <= 0.85
    assert report.threshold == bundle.threshold
    assert any("Threshold" in n for n in report.notes), "notes should document threshold selection"


def test_no_leakage_between_threshold_and_report_folds(trained) -> None:
    """Threshold-fold F1 must not wildly exceed test F1 — that gap was the leakage signature.

    An earlier version calibrated on a fold that *contained* the threshold fold and reported
    0.955 threshold-fold F1 against 0.802 test F1. A sane generalisation gap is under ~0.15.
    """
    _, report = trained
    note = next(n for n in report.notes if "threshold-fold F1=" in n)
    thr_f1 = float(note.split("threshold-fold F1=")[1].rstrip(")."))
    gap = thr_f1 - report.scores["f1"]
    assert gap < 0.15, (
        f"threshold-fold F1 {thr_f1:.3f} vs test F1 {report.scores['f1']:.3f} "
        f"(gap {gap:.3f}) suggests the folds are not disjoint"
    )


def test_smote_was_applied_after_the_split(trained) -> None:
    _, report = trained
    smote_notes = [n for n in report.notes if n.startswith("SMOTE:")]
    assert smote_notes, "SMOTE step not recorded in notes"
    assert any("only" in n and "split" in n for n in report.notes)


def test_confusion_matrix_is_internally_consistent(trained) -> None:
    _, report = trained
    c = report.confusion
    assert c["tn"] + c["fp"] + c["fn"] + c["tp"] == report.n_samples
    assert report.scores["recall"] == pytest.approx(c["tp"] / (c["tp"] + c["fn"]), abs=1e-6)
    assert report.scores["precision"] == pytest.approx(c["tp"] / (c["tp"] + c["fp"]), abs=1e-6)


# --- prediction --------------------------------------------------------------


def _record(**overrides) -> dict:
    base = {f: 0.0 for f in FEATURES}
    base.update(
        account_length=120,
        number_vmail_messages=0,
        total_day_minutes=180,
        total_day_calls=110,
        total_eve_minutes=200,
        total_eve_calls=100,
        total_night_minutes=200,
        total_night_calls=100,
        total_intl_minutes=10,
        total_intl_calls=4,
        number_customer_service_calls=1,
        international_plan=0,
        voice_mail_plan=0,
    )
    base.update(overrides)
    return base


def test_predict_returns_a_valid_prediction(module) -> None:
    bundle = module._bundle
    pred = module.predict(_record())
    assert isinstance(pred, Prediction)
    assert pred.module_id == "churn"
    assert 0.0 <= pred.score <= 1.0
    assert pred.label and pred.decision
    assert pred.factors, "prediction should inline its top drivers"
    assert pred.meta["threshold"] == bundle.threshold


def test_missing_feature_raises_value_error(trained) -> None:
    bundle, _ = trained
    with pytest.raises(ValueError, match="Missing required features"):
        bundle.frame({"account_length": 10})


def test_high_risk_profile_scores_above_low_risk_profile(trained) -> None:
    """Domain sanity: the known churn levers must move the score the right way."""
    bundle, _ = trained
    quiet = bundle.predict_proba(_record(number_customer_service_calls=0, international_plan=0))
    angry = bundle.predict_proba(
        _record(number_customer_service_calls=6, international_plan=1, number_vmail_messages=0)
    )
    assert angry > quiet, f"angry customer ({angry:.3f}) should outscore quiet ({quiet:.3f})"


def test_voice_mail_is_protective_when_isolated(module) -> None:
    """Real data: 16.7% churn without voice mail vs 8.7% with.

    Earlier this test moved ``voice_mail_plan`` *and* ``number_vmail_messages`` together, which
    confounded the comparison — the message count carries its own effect and swamped the plan
    flag, making a genuinely protective feature look harmful. Vary one lever at a time.
    """
    bundle = module._bundle
    base = _record()
    flipped = _record(voice_mail_plan=1)  # vmail messages held at 0

    assert bundle.predict_proba(flipped) < bundle.predict_proba(base), (
        f"voice mail should lower risk: with={bundle.predict_proba(flipped):.4f} "
        f"without={bundle.predict_proba(base):.4f}"
    )


def test_population_level_scores_track_the_real_churn_rates(module) -> None:
    """The claim that actually matters: group-level calibration on the real 3,333 customers."""
    import pandas as pd

    bundle = module._bundle
    df, _ = load_telco()
    scores = bundle.model.predict_proba(df[list(FEATURES)])[:, 1]

    predicted = pd.Series(scores).groupby(df["voice_mail_plan"].values).mean()
    actual = df.groupby("voice_mail_plan")["churn"].mean()

    # Direction must match...
    assert predicted.idxmin() == actual.idxmin(), "voice mail group ordering disagrees with reality"
    # ...and magnitude must be in the right ballpark (within 4 percentage points).
    for group in actual.index:
        assert abs(predicted[group] - actual[group]) < 0.04, (
            f"group {group}: predicted {predicted[group]:.4f} vs actual {actual[group]:.4f}"
        )


def test_scores_are_spread_across_the_real_distribution(module) -> None:
    """Regression test for the isotonic defect — measured on real data, not a synthetic ladder.

    The original bug collapsed every low-risk customer onto exactly 0.0208. Checking a ladder
    with all other features pinned is the wrong probe: gradient boosting is piecewise constant,
    so a fixed-feature ladder *legitimately* repeats values (measured margins run
    -1.1856 x3, -0.6997, +0.5066 x3). The meaningful question is whether the score separates
    the real population.
    """
    import numpy as _np

    bundle = module._bundle
    df, _ = load_telco()
    scores = bundle.model.predict_proba(df[list(FEATURES)])[:, 1]

    assert len(_np.unique(_np.round(scores, 6))) > 200, "score distribution has collapsed"
    assert scores.max() - scores.min() > 0.5, f"score range too narrow: {scores.min():.4f}-{scores.max():.4f}"
    # The four quantiles must be genuinely distinct, not stacked on one constant.
    q = _np.quantile(scores, [0.1, 0.5, 0.9, 0.99])
    assert len({round(float(v), 6) for v in q}) == 4, f"quantiles collapsed: {q}"


def test_calibrated_score_is_monotone_in_the_explained_margin(module) -> None:
    """The invariant that ties the score to the explanation.

    With ``cv=3`` and no frozen estimator the calibrator refit the trees, so the shipped model
    was *not* the model SHAP explained — which produced real inversions (voice mail looked
    protective on the margin but riskier after calibration). A strictly monotone map is what
    makes "these bars explain this score" a true statement.
    """
    import pandas as pd

    bundle = module._bundle
    df, _ = load_telco()
    sample = df.sample(300, random_state=3)

    margins, scores = [], []
    for _, row in sample.iterrows():
        rec = {f: float(row[f]) for f in FEATURES}
        X = bundle.frame(rec)
        margins.append(float(bundle.raw_model.decision_function(X)[0]))
        scores.append(float(bundle.model.predict_proba(X)[0, 1]))

    order = np.argsort(margins)
    ranked = np.array(scores)[order]
    # Allow ties (trees are piecewise constant) but never a decrease.
    violations = int((np.diff(ranked) < -1e-9).sum())
    assert violations == 0, f"{violations} monotonicity violations: calibration drifted from the explained model"


# --- explanation -------------------------------------------------------------


def test_shap_attributions_sum_to_the_raw_margin(module) -> None:
    """The additive property is what makes SHAP an explanation and not a decoration.

    TreeSHAP is exact in the model's raw log-odds, so base + sum(attributions) must reproduce
    ``decision_function`` — *not* the calibrated probability. Asserting against the calibrated
    score would pass only by coincidence, and would hide the fact that the UI must label the
    axis as log-odds.
    """
    bundle = module._bundle
    payload = _record(number_customer_service_calls=5, international_plan=1)
    X = bundle.frame(payload)

    attributions, base = bundle.shap_values(payload)
    reconstructed = base + float(attributions.sum())
    margin = float(bundle.raw_model.decision_function(X)[0])

    assert reconstructed == pytest.approx(margin, abs=1e-6), (
        f"SHAP sum {reconstructed:.6f} != raw margin {margin:.6f}"
    )


def test_explanation_declares_its_attribution_space(module) -> None:
    """Guards against implying the bars sum to the displayed probability when they do not."""
    exp = module.explain(_record(number_customer_service_calls=7))
    assert exp.space == "log_odds"
    assert any("do NOT sum to the calibrated" in c for c in exp.caveats)

    # And the caveat must be true, not decorative: the two quantities genuinely differ.
    payload = _record(number_customer_service_calls=7)
    _, base = module._bundle.shap_values(payload)
    margin = base + float(sum(a.attribution for a in exp.attributions))
    score = module.predict(payload).score
    assert 0.0 <= score <= 1.0
    assert margin != pytest.approx(score, abs=1e-3), (
        "if margin == score the space label is meaningless; re-check the calibrator"
    )


def test_explain_returns_sorted_attributions(module) -> None:
    exp = module.explain(_record(number_customer_service_calls=7))
    assert isinstance(exp, Explanation)
    assert exp.kind == "shap"
    assert len(exp.attributions) == len(FEATURES)
    mags = [abs(a.attribution) for a in exp.attributions]
    assert mags == sorted(mags, reverse=True), "attributions must be sorted by |attribution|"
    assert exp.summary
    assert exp.caveats


def test_service_calls_are_a_strong_positive_driver_for_an_angry_customer(module) -> None:
    """Empirical churn jumps from ~10% at 0-3 calls to ~46-64% at 4-6 calls.

    Asserting that service calls are *the* single largest bar is fragile — other features can
    carry larger offsetting attributions for one record. What is robust, and what the UI
    promises, is that this lever pushes hard in the right direction.
    """
    exp = module.explain(_record(number_customer_service_calls=8, international_plan=1))
    by_name = {a.feature: a for a in exp.attributions}

    calls = by_name["number_customer_service_calls"]
    assert calls.direction == "increases_risk"
    assert calls.attribution > 0.5, f"expected a strong positive push, got {calls.attribution:+.4f}"

    intl = by_name["international_plan"]
    assert intl.direction == "increases_risk", "international plan is a known churn lever (42% vs 11%)"


# --- tools -------------------------------------------------------------------


def test_rank_at_risk_customers(module) -> None:
    result = module.tool_rank_at_risk_customers(n=5, min_score=0.0)
    assert result["n_returned"] == 5
    assert result["n_scored"] == 3333
    scores = [c["score"] for c in result["customers"]]
    assert scores == sorted(scores, reverse=True), "must be ranked, not just filtered"
    assert all(c["decision"] for c in result["customers"])


def test_rank_respects_min_score(module) -> None:
    result = module.tool_rank_at_risk_customers(n=10, min_score=0.9)
    assert all(c["score"] >= 0.9 for c in result["customers"])
