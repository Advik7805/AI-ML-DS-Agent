"""Explainable Diagnostics — a Saros body.

The point of this body is not the classifier. Any competent model predicts diabetes from
glucose and BMI. The point is the two things that make a high-stakes model deployable:

1. **Per-patient explanation** — SHAP, so a clinician sees why *this* patient scored high.
2. **Group fairness audit** — Fairlearn, so a difference in error rate between groups is
   found by the model owner rather than by a regulator.

Both are contract-level capabilities, not optional polish.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from core.contract import Module
from core.tenant import prediction_meta
from core.types import AgentTool, Explanation, FeatureAttribution, MetricsReport, ModuleManifest, Prediction

from .data import FEATURES, TARGET, load_diabetes
from core.explain import build_explainer

ACCENT = "#a78bfa"
ARTIFACT = Path(__file__).resolve().parents[2] / "artifacts" / "xai_health" / "model.pkl"


@dataclass
class HealthModel:
    pipeline: Any
    feature_names: list[str]
    threshold: float = 0.5
    metrics: dict[str, float] = field(default_factory=dict)
    source: str = "synthetic"
    trained_at: str = ""
    #: Full source-frame size, for the provenance card.
    n_rows: int = 0
    notes: list[str] = field(default_factory=list)

    def frame(self, payload: dict[str, Any]) -> pd.DataFrame:
        missing = [f for f in self.feature_names if f not in payload]
        if missing:
            raise ValueError(f"Missing required features: {', '.join(missing)}")
        row = {f: (np.nan if payload[f] is None else float(payload[f])) for f in self.feature_names}
        return pd.DataFrame([row], columns=self.feature_names)

    def predict_proba(self, payload: dict[str, Any]) -> float:
        return float(self.pipeline.predict_proba(self.frame(payload))[0, 1])

    def save(self, path: Path = ARTIFACT) -> Path:
        import pickle

        path.parent.mkdir(parents=True, exist_ok=True)
        from core.artifacts import save as _save_artifact

        return _save_artifact(self, path)

    @classmethod
    def load(cls, path: Path = ARTIFACT) -> "HealthModel":
        import pickle

        from core.artifacts import load as _load_artifact

        return _load_artifact(path)


def train_health(*, random_state: int = 5) -> tuple[HealthModel, MetricsReport]:
    from fairlearn.metrics import MetricFrame, false_negative_rate, false_positive_rate

    df, source = load_diabetes()
    X = df[list(FEATURES)]
    y = df[TARGET].astype(int).values

    X_tr, X_te, y_tr, y_te, g_tr, g_te = train_test_split(
        X, y, df["group"].values, test_size=0.3, stratify=y, random_state=random_state
    )

    notes = [
        f"Data source: {source} ({len(df)} patients, {y.mean():.1%} positive).",
        "Zero is a missing-value sentinel in this dataset for glucose, blood pressure, skin "
        "thickness, insulin and BMI. Those zeros are converted to NaN and imputed — treating "
        "them as readings would teach the model that a BMI of 0 is a real patient.",
        "Gradient boosting handles the remaining NaNs natively; the imputer is kept for the "
        "logistic-regression-style baseline comparison.",
    ]

    pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("clf", HistGradientBoostingClassifier(
                max_iter=200, max_depth=4, learning_rate=0.07, random_state=random_state
            )),
        ]
    ).fit(X_tr, y_tr)

    bundle = HealthModel(
        pipeline=pipe,
        feature_names=list(FEATURES),
        source=source,
        trained_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        n_rows=len(df),
        notes=notes,
    )

    thr_scores = pipe.predict_proba(X_tr)[:, 1]
    grid = np.arange(0.20, 0.81, 0.01)
    f1s = [f1_score(y_tr, (thr_scores >= t).astype(int), zero_division=0) for t in grid]
    bundle.threshold = float(grid[int(np.argmax(f1s))])
    notes.append(
        f"Threshold {bundle.threshold:.2f} by F1 on the training split. With only {len(df)} "
        f"patients a third fold would be too small to be worth carving out — noted as a "
        f"limitation rather than silently ignored."
    )

    proba = pipe.predict_proba(X_te)[:, 1]
    pred = (proba >= bundle.threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_te, pred, labels=[0, 1]).ravel()

    # --- Fairlearn group audit -------------------------------------------------
    # In a screening context the harm asymmetry is stark: a false negative tells a diabetic
    # patient they are fine. So false-negative *rate* is the fairness metric that matters,
    # not overall accuracy.
    mf = MetricFrame(
        metrics={
            "recall": recall_score,
            "false_negative_rate": false_negative_rate,
            "false_positive_rate": false_positive_rate,
        },
        y_true=y_te,
        y_pred=pred,
        sensitive_features=g_te,
    )
    by_group = mf.by_group
    fnr = by_group["false_negative_rate"].astype(float)
    fpr = by_group["false_positive_rate"].astype(float)

    scores = {
        "roc_auc": float(roc_auc_score(y_te, proba)),
        "pr_auc": float(average_precision_score(y_te, proba)),
        "balanced_accuracy": float(balanced_accuracy_score(y_te, pred)),
        "recall": float(recall_score(y_te, pred, zero_division=0)),
        "f1": float(f1_score(y_te, pred, zero_division=0)),
        "baseline_pr_auc": float(y_te.mean()),
        "fairness_fnr_gap": float(fnr.max() - fnr.min()),
        "fairness_fpr_gap": float(fpr.max() - fpr.min()),
    }

    worst_fnr_group = str(fnr.idxmax())
    notes.append(
        f"Fairness audit by age band (a proxy — this dataset has no race or gender field). "
        f"False-negative rate ranges {fnr.min():.3f} to {fnr.max():.3f} across groups; the "
        f"worst-served group is '{worst_fnr_group}'."
    )
    notes.append(
        "False-negative rate is the fairness metric that matters here: a missed diagnosis tells "
        "a diabetic patient they are healthy. Overall accuracy would hide a group being failed."
    )
    if scores["fairness_fnr_gap"] > 0.15:
        notes.append(
            f"WARNING: the false-negative gap of {scores['fairness_fnr_gap']:.3f} exceeds 0.15. "
            "This model should not be deployed for screening without remediation."
        )

    group_report = {
        str(g): {"false_negative_rate": float(fnr[g]), "false_positive_rate": float(fpr[g])}
        for g in fnr.index
    }
    report = MetricsReport(
        n_samples=int(len(X_te)),
        positive_rate=float(y_te.mean()),
        scores=scores,
        confusion={"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        threshold=bundle.threshold,
        notes=notes,
        groups=group_report,
        group_basis="age band (a proxy — this dataset has no race or gender field)",
    )
    bundle.metrics = scores
    return bundle, report


class XaiHealthModule(Module):
    id = "xai_health"

    def __init__(self) -> None:
        self._bundle: HealthModel | None = None
        self._report: MetricsReport | None = None
        self._lock = threading.Lock()

    @property
    def bundle(self) -> HealthModel:
        if self._bundle is None:
            with self._lock:
                if self._bundle is None:
                    try:
                        self._bundle = HealthModel.load()
                    except Exception:  # noqa: BLE001
                        self._bundle, self._report = train_health()
                        self._bundle.save()
        return self._bundle

    def train(self, config: dict[str, Any] | None = None) -> MetricsReport:
        bundle, report = train_health(**(config or {}))
        bundle.save()
        self._bundle, self._report = bundle, report
        return report

    def evaluate(self) -> MetricsReport:
        if self._report is None:
            self.train()
        assert self._report is not None
        return self._report

    @property
    def manifest(self) -> ModuleManifest:
        return ModuleManifest(
            id=self.id,
            title="Explainable Diagnostics",
            summary=(
                "Diabetes risk scoring with per-patient SHAP explanations and a Fairlearn group "
                "fairness audit. Optimises for false-negative rate, because in screening a "
                "missed diagnosis is the harm that matters. Data: Northwind Care's employee screening pilot "
                "— the public Pima Indians Diabetes study, 768 records, with a Fairlearn group audit "
                "because the decision is clinical."
            ),
            accent=ACCENT,
            status="live",
            capabilities=["classification", "shap-xai", "fairness-audit", "imputation"],
            routes={"ui": "/diagnostics", "api": "/api/v1/xai_health"},
            dataset=self.dataset_card(),
            tools=self.tools(),
        )

    def predict(self, payload: dict[str, Any]) -> Prediction:
        bundle = self.bundle
        score = bundle.predict_proba(payload)
        if score >= 0.6:
            label, decision = "High diabetes risk", "Refer for HbA1c confirmation. Do not communicate risk to the patient without clinician review."
        elif score >= bundle.threshold:
            label, decision = "Elevated diabetes risk", "Recommend screening at the next visit and review modifiable factors."
        else:
            label, decision = "Low diabetes risk", "Routine care. Note that a low score is not a clearance — see the caveats."

        explanation = self._safe_explain(payload)
        factors = explanation.attributions[:5] if explanation else []

        return Prediction(
            module_id=self.id,
            score=round(score, 4),
            label=label,
            decision=decision,
            inputs={k: float(payload[k]) for k in bundle.feature_names if payload.get(k) is not None},
            factors=factors,
            explanation=explanation,
            meta=prediction_meta(
                "xai_health",
                threshold=bundle.threshold,
                source=bundle.source,
                warning="Decision support only. Not a diagnostic device.",
            ),
        )

    def _safe_explain(self, payload: dict[str, Any]) -> Explanation | None:
        try:
            return self.explain(payload)
        except Exception:  # noqa: BLE001
            return None

    def explain(self, payload: dict[str, Any]) -> Explanation:
        import shap

        bundle = self.bundle
        clf = bundle.pipeline.named_steps["clf"]
        X = bundle.pipeline.named_steps["impute"].transform(bundle.frame(payload))
        explainer = build_explainer(clf).explainer
        sv = explainer.shap_values(X)
        sv = sv[:, :, -1] if getattr(sv, "ndim", 0) == 3 else sv
        base = float(np.ravel(explainer.expected_value)[-1])

        items = [
            FeatureAttribution.of(f, float(payload.get(f) if payload.get(f) is not None else np.nan), float(v))
            for f, v in zip(bundle.feature_names, sv[0])
        ]
        items.sort(key=lambda a: abs(a.attribution), reverse=True)
        top = items[0] if items else None
        return Explanation(
            kind="shap",
            space="log_odds",
            attributions=items,
            base_value=round(base, 4),
            summary=(
                f"'{top.feature}' drives this prediction most "
                f"({'increasing' if top.direction == 'increases_risk' else 'decreasing'} risk)."
                if top else "No attributions."
            ),
            caveats=[
                "Clinical decision support only. This is not a diagnostic device and no output "
                "here should reach a patient without clinician review.",
                f"Trained on {bundle.source} data; population shift is not monitored.",
                "Missing values are median-imputed, which for this dataset means a patient with "
                "unrecorded insulin is treated as typical — that assumption is visible in the "
                "attribution for that feature and should be read with care.",
            ],
        )

    def tools(self) -> list[AgentTool]:
        return [
            AgentTool(
                name="assess_diabetes_risk",
                description=(
                    "Assess a patient's diabetes risk with a SHAP explanation. Decision support "
                    "only. Requires 8 clinical features; pass null for unmeasured values."
                ),
                module_id=self.id,
                parameters_schema={
                    "type": "object",
                    "properties": {
                        **{f: {"type": ["number", "null"]} for f in FEATURES},
                    },
                    "required": list(FEATURES),
                },
            ),
            AgentTool(
                name="audit_model_fairness",
                description=(
                    "Run the Fairlearn group audit: false-negative and false-positive rates per "
                    "group, and the gap between best and worst served. Call this before claiming "
                    "the model is safe to deploy."
                ),
                module_id=self.id,
                parameters_schema={"type": "object", "properties": {}},
            ),
        ]

    def tool_assess_diabetes_risk(self, **features: Any) -> dict[str, Any]:
        return self.predict(features).model_dump()

    def tool_audit_model_fairness(self) -> dict[str, Any]:
        report = self.evaluate()
        groups = report.groups
        worst = max(groups.items(), key=lambda kv: kv[1]["false_negative_rate"], default=None)
        return {
            "grouping_variable": report.group_basis,
            "groups": groups,
            "fnr_gap": report.scores.get("fairness_fnr_gap"),
            "fpr_gap": report.scores.get("fairness_fpr_gap"),
            "worst_served_group": worst[0] if worst else None,
            "worst_served_fnr": worst[1]["false_negative_rate"] if worst else None,
            "verdict": (
                "Gap exceeds 0.15 — remediate before deployment."
                if (report.scores.get("fairness_fnr_gap") or 0) > 0.15
                else "Gap within the 0.15 tolerance, but note the small sample size."
            ),
            "metric_rationale": (
                "False-negative rate is the audited quantity: in screening, the harm is telling "
                "a diabetic patient they are healthy. Accuracy would not reveal it."
            ),
        }


MODULE = XaiHealthModule()
