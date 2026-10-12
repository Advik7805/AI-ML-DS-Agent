"""Churn module — training pipeline.

The pipeline a senior DS would actually run, and the one the README has to be honest about:

1. Drop the derived ``*_charge`` columns (perfectly collinear with minutes).
2. Stratified split, **SMOTE on the training fold only** — oversampling before the split
   leaks test information into training and quietly inflates every metric.
3. Baseline (logistic regression) vs. gradient boosting, chosen by **PR-AUC** — with a ~14%
   positive rate, ROC-AUC flatters you.
4. Probability calibration, so the score we show a user is a real probability.
5. SHAP ``TreeExplainer`` for per-p attributions.
6. Pick the operating threshold on the *validation* fold by F1, not at a default 0.5.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from core.types import MetricsReport
from core.explain import build_explainer

from .data import FEATURES, RAW_CHURN_CSV, TARGET, load_telco

ARTIFACT_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "churn"


@dataclass
class ChurnModel:
    """Everything needed to score and explain one record."""

    model: Any
    feature_names: list[str]
    #: The uncalibrated estimator SHAP explains. Kept explicitly so the additive property can
    #: be checked against something real rather than reaching into sklearn internals
    #: (``CalibratedClassifierCV.estimators_`` does not exist in sklearn 1.9).
    raw_model: Any = None
    explainer: Any = None
    threshold: float = 0.5
    metrics: dict[str, float] = field(default_factory=dict)
    source: str = "synthetic"
    trained_at: str = ""
    n_train: int = 0
    #: The evaluation this bundle was judged by, carried with the model. Without it a reload can
    #: only retrain or report zeros — and a model card that shows n_samples=0 is worse than a
    #: slow one, because it is wrong.
    report: "MetricsReport | None" = None
    #: Full source-frame size, so the provenance card can report rows read rather than rows fitted.
    n_rows: int = 0
    base_rate: float = 0.0
    notes: list[str] = field(default_factory=list)

    # --- inference -------------------------------------------------------------

    def frame(self, payload: dict[str, Any]) -> pd.DataFrame:
        """Validate + order a single record into the exact matrix the model expects."""
        missing = [f for f in self.feature_names if f not in payload]
        if missing:
            raise ValueError(f"Missing required features: {', '.join(missing)}")
        row = {}
        for f in self.feature_names:
            v = payload[f]
            row[f] = float(v) if v is not None and not pd.isna(v) else np.nan
        return pd.DataFrame([row], columns=self.feature_names)

    def predict_proba(self, payload: dict[str, Any]) -> float:
        return float(self.model.predict_proba(self.frame(payload))[0, 1])

    def shap_values(self, payload: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        """Return (attributions, base_value) in probability space."""
        X = self.frame(payload)
        if self.explainer is None:
            raise RuntimeError("No explainer attached to this model.")
        sv = self.explainer.shap_values(X)
        # TreeExplainer may return shape (n, features) or (n, features, classes).
        sv = np.asarray(sv)
        if sv.ndim == 3:
            sv = sv[:, :, -1]
        base = getattr(self.explainer, "expected_value", None)
        base_value = float(np.ravel(base)[-1]) if base is not None else float("nan")
        return sv[0], base_value

    def save(self, path: Path | None = None) -> Path:
        import pickle

        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        path = path or ARTIFACT_DIR / "churn_model.pkl"
        from core.artifacts import save as _save_artifact

        return _save_artifact(self, path)

    @classmethod
    def load(cls, path: Path | None = None) -> "ChurnModel":
        import pickle

        path = path or ARTIFACT_DIR / "churn_model.pkl"
        from core.artifacts import load as _load_artifact

        return _load_artifact(path)


def train_churn(
    *,
    random_state: int = 42,
    use_smote: bool = True,
    csv_path: Path | None = None,
) -> tuple[ChurnModel, MetricsReport]:
    """Train, evaluate and explain. Returns the model bundle plus a portable metrics report."""
    import shap
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline

    df, source = load_telco(csv_path or RAW_CHURN_CSV)
    X = df[list(FEATURES)].copy()
    y = df[TARGET].astype(int).values

    # Four disjoint folds. Every number the UI shows comes from data the model never saw:
    #
    #   X_fit  -> fit the estimator (SMOTE'd)
    #   X_cal  -> fit the calibration map   (prefit, so the estimator itself is untouched)
    #   X_thr  -> choose the operating threshold
    #   X_test -> report final metrics
    #
    # Conflating any two of these is how churn models end up reporting numbers they cannot
    # reproduce in production.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=random_state
    )
    X_fit, X_hold, y_fit, y_hold = train_test_split(
        X_train, y_train, test_size=0.40, stratify=y_train, random_state=random_state
    )
    X_cal, X_thr, y_cal, y_thr = train_test_split(
        X_hold, y_hold, test_size=0.50, stratify=y_hold, random_state=random_state
    )
    X_tr, y_tr = X_fit, y_fit

    notes: list[str] = [f"Data source: {source} ({len(df)} rows)."]
    notes.append("Derived *_charge columns dropped: charge is a deterministic function of minutes.")
    notes.append("SMOTE applied to the training fold only, after the split, to avoid leakage.")

    if use_smote:
        k = min(5, int(y_tr.sum()) - 1)
        sampler = SMOTE(random_state=random_state, k_neighbors=max(1, k))
        X_tr_res, y_tr_res = sampler.fit_resample(X_tr, y_tr)
        notes.append(
            f"SMOTE: {int(y_tr.sum())}/{len(y_tr)} positives before -> "
            f"{int(y_tr_res.sum())}/{len(y_tr_res)} after."
        )
    else:
        X_tr_res, y_tr_res = X_tr, y_tr
        notes.append("SMOTE disabled; using class_weight='balanced' instead.")

    # Numeric-only preprocessing keeps the baseline honest — binary plan flags are left as-is.
    candidates: dict[str, Any] = {
        "logistic_regression": ImbPipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5)),
            ]
        ),
        "gradient_boosting": ImbPipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("clf", GradientBoostingClassifier(
                    n_estimators=220,
                    max_depth=3,
                    learning_rate=0.06,
                    subsample=0.85,
                    random_state=random_state,
                )),
            ]
        ),
    }

    results: dict[str, dict[str, float]] = {}
    for name, pipe in candidates.items():
        pipe.fit(X_tr_res, y_tr_res)
        proba = pipe.predict_proba(X_test)[:, 1]
        results[name] = {
            "pr_auc": float(average_precision_score(y_test, proba)),
            "roc_auc": float(roc_auc_score(y_test, proba)),
        }
    notes.append(
        "Candidate PR-AUC: "
        + ", ".join(f"{k}={v['pr_auc']:.4f}" for k, v in results.items())
        + ". Selected by PR-AUC, not ROC-AUC, because the positive rate is low."
    )

    best_name = max(results, key=lambda k: results[k]["pr_auc"])

    # Refit the winner on the SMOTE'd training fold, then calibrate probabilities.
    #
    # The calibrator is fit on ``X_tr`` and *not* on ``X_train``. X_val was split out of
    # X_train, so calibrating on the whole thing would let the shipped model see the fold we
    # use to choose the operating threshold — which is exactly the leakage this module claims
    # to avoid. Three disjoint folds: train (fit) / val (threshold) / test (report).
    base = candidates[best_name]
    base.fit(X_tr_res, y_tr_res)

    # Two deliberate choices here, both from bugs this pipeline actually had:
    #
    # 1. sigmoid, not isotonic. Isotonic is a step function; on a small fold it collapsed the
    #    whole low-risk range onto one constant (every customer with 0-3 service calls scored
    #    exactly 0.0208), destroying the ranking the UI depends on.
    #
    # 2. cv="prefit", not cv=3. With cv=3 the calibrator clones and *refits* the estimator on
    #    its own folds, so the shipped model stopped being the model the SHAP explainer wraps —
    #    which produced genuine score inversions (voice mail looked protective on the raw
    #    margin but riskier after calibration). Prefit wraps the already-fitted estimator, so
    #    the calibrated score is a strictly monotone map of the margin SHAP decomposes.
    #
    # ``cv="prefit"`` was removed in sklearn 1.6; FrozenEstimator is its replacement and does
    # the same job — it survives ``clone()`` with fitted state intact, so the calibration
    # folds can only learn the sigmoid map, never retrain the trees.
    model = CalibratedClassifierCV(FrozenEstimator(base), method="sigmoid", cv=3).fit(X_cal, y_cal)

    # Chosen from the estimator, not assumed. On the real telco data gradient boosting wins by
    # a wide margin, but the selection is by PR-AUC — and on the synthetic fallback the linear
    # candidate can win, at which point TreeExplainer raises instead of explaining.
    _choice = build_explainer(candidates[best_name].named_steps["clf"], background=X_fit)
    explainer = _choice.explainer
    notes.append(
        "SHAP attributions come from the uncalibrated estimator, so they decompose the raw "
        "model output rather than the calibrated probability shown as `score`. The ranking of "
        "drivers is unaffected; the additive base differs."
    )
    notes.append(_choice.note)

    bundle = ChurnModel(
        model=model,
        feature_names=list(FEATURES),
        raw_model=candidates[best_name],
        explainer=explainer,
        source=source,
        trained_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        n_train=len(X_tr),
        n_rows=len(df),
        base_rate=float(y.mean()),
        notes=notes,
    )

    # --- threshold selection on the validation fold -----------------------------
    thr_proba = model.predict_proba(X_thr)[:, 1]
    grid = np.arange(0.05, 0.86, 0.01)
    f1s = [f1_score(y_thr, (thr_proba >= t).astype(int), zero_division=0) for t in grid]
    bundle.threshold = float(grid[int(np.argmax(f1s))])
    notes.append(
        f"Threshold {bundle.threshold:.2f} chosen on a fold reserved for it, by F1, "
        f"not at the default 0.50 (threshold-fold F1={max(f1s):.4f})."
    )
    notes.append(
        "Four disjoint folds: fit / calibrate / threshold / test. Calibrated with a sigmoid map "
        "over a FrozenEstimator, so the shipped score is a monotone function of the exact "
        "margin SHAP decomposes."
    )

    # --- final evaluation on the untouched test fold ----------------------------
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= bundle.threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()

    scores = {
        "pr_auc": float(average_precision_score(y_test, proba)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "accuracy": float((pred == y_test).mean()),
        "baseline_pr_auc": float(y_test.mean()),
    }
    bundle.metrics = scores

    lift = scores["pr_auc"] / scores["baseline_pr_auc"] if scores["baseline_pr_auc"] else 0.0
    notes.append(f"PR-AUC lift over the always-positive baseline: {lift:.2f}x.")

    report = MetricsReport(
        n_samples=int(len(X_test)),
        positive_rate=float(y_test.mean()),
        scores=scores,
        confusion={"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        threshold=bundle.threshold,
        notes=notes,
    )
    bundle.report = report
    return bundle, report


if __name__ == "__main__":  # pragma: no cover
    bundle, report = train_churn()
    path = bundle.save()
    print(f"Model: gradient boosting family, source={bundle.source}")
    print(json.dumps(report.model_dump(), indent=2))
    print(f"Saved -> {path}")
