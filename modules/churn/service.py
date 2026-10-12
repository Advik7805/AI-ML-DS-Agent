"""Churn module — the body itself.

Implements :class:`core.contract.Module`. This is the shape every other Saros body copies:
``manifest`` for the hub, ``predict`` for scoring, ``explain`` for XAI, ``tools`` for the
agent, ``evaluate`` for the metrics card.

The model trains lazily on first use and caches to ``artifacts/churn/`` so a cold API
container does not retrain per request.
"""

from __future__ import annotations

import threading
from typing import Any

from core.contract import Module
from core.tenant import prediction_meta
from core.types import AgentTool, Explanation, FeatureAttribution, MetricsReport, ModuleManifest, Prediction

from .data import FEATURES, TARGET, load_telco
from .train import ChurnModel, train_churn

ACCENT = "#f59e0b"


def _band(score: float, threshold: float) -> tuple[str, str]:
    """Map a probability to a label and a decision a retention team can act on."""
    if score >= max(threshold + 0.2, 0.75):
        return "Critical churn risk", "Escalate to save desk within 24h; offer contract buyout or a discount with a 6-month term."
    if score >= threshold:
        return "High churn risk", "Trigger a retention offer: targeted discount plus a proactive support check-in this week."
    if score >= max(threshold - 0.15, 0.2):
        return "Elevated churn risk", "Add to the nurture segment; monitor service-call volume and re-score in 30 days."
    return "Low churn risk", "No action. Exclude from retention spend to protect margin."


class ChurnModule(Module):
    id = "churn"

    def __init__(self) -> None:
        self._bundle: ChurnModel | None = None
        self._report: MetricsReport | None = None
        self._lock = threading.Lock()

    # --- lifecycle -------------------------------------------------------------

    @property
    def bundle(self) -> ChurnModel:
        """Load the cached artifact, training it on first use."""
        if self._bundle is None:
            with self._lock:
                if self._bundle is None:
                    try:
                        self._bundle = ChurnModel.load()
                    except Exception:  # noqa: BLE001 - no artifact yet, or stale pickle
                        self._bundle, self._report = train_churn()
                        self._bundle.save()
        return self._bundle

    def train(self, config: dict[str, Any] | None = None) -> MetricsReport:
        config = config or {}
        bundle, report = train_churn(**config)
        bundle.save()
        self._bundle, self._report = bundle, report
        return report

    def evaluate(self) -> MetricsReport:
        if self._report is None:
            # Retraining is the only way to get metrics; the artifact does not store them.
            self.train()
        assert self._report is not None
        return self._report

    # --- contract --------------------------------------------------------------

    @property
    def manifest(self) -> ModuleManifest:
        return ModuleManifest(
            id=self.id,
            title="Churn & Retention",
            summary=(
                "Gradient-boosted churn scoring with SMOTE for class imbalance, isotonic "
                "calibration, and SHAP explanations. Ranks an entire book of business by "
                "attrition risk and recommends a retention action per customer. Data: Northwind Connect's "
                "subscriber billing extract — the public IBM Telco Customer Churn set, 3,333 rows, "
                "keyed by customer_id and joined to the fraud body."
            ),
            accent=ACCENT,
            status="live",
            capabilities=["classification", "imbalance-handling", "calibration", "shap-xai", "ranking"],
            routes={"ui": "/churn", "api": "/api/v1/churn"},
            accepts=self.upload_spec().get("accepts", ""),
            dataset=self.dataset_card(),
            tools=self.tools(),
        )

    def upload_spec(self) -> dict[str, Any]:
        """What a subscriber extract from your side needs to look like."""
        from .data import BINARY_FEATURES, NUMERIC_FEATURES, TARGET

        return {
            "kind": "csv",
            "accepts": (
                "CSV, one row per customer. 13 model columns (11 numeric + international_plan, "
                "voice_mail_plan as 0/1 or yes/no). Optional: state, area_code, *_charge columns, "
                "and a 'churn' label to score the model against your own outcomes."
            ),
            "required": [*NUMERIC_FEATURES, *BINARY_FEATURES],
            "optional": ["state", "area_code", "customer_id", "total_day_charge", "total_eve_charge",
                         "total_night_charge", "total_intl_charge", TARGET],
            "features": [*NUMERIC_FEATURES, *BINARY_FEATURES],
            "label": TARGET,
            "aliases": {
                TARGET: ["churned", "is_churn", "churn_flag"],
                "number_customer_service_calls": ["service_calls", "custserv_calls"],
                "international_plan": ["intl_plan"],
                "voice_mail_plan": ["vmail_plan"],
                "total_day_minutes": ["day_mins", "total_day_mins"],
                "total_eve_minutes": ["eve_mins"],
                "total_night_minutes": ["night_mins"],
                "total_intl_minutes": ["intl_mins"],
            },
            "notes": [
                "Columns are matched case- and punctuation-insensitively: 'Total Day Minutes' "
                "works as well as 'total_day_minutes'.",
            ],
        }

    def predict(self, payload: dict[str, Any]) -> Prediction:
        bundle = self.bundle
        try:
            X = bundle.frame(payload)
        except ValueError as exc:
            raise ValueError(str(exc)) from None

        score = float(bundle.model.predict_proba(X)[0, 1])
        explanation = self._safe_explain(payload)
        factors = explanation.attributions[:5] if explanation else []
        label, decision = _band(score, bundle.threshold)

        return Prediction(
            module_id=self.id,
            score=round(score, 4),
            label=label,
            decision=decision,
            inputs={k: float(payload[k]) for k in FEATURES if payload.get(k) is not None},
            factors=factors,
            explanation=explanation,
            meta=prediction_meta(
                "churn",
                threshold=bundle.threshold,
                base_rate=round(bundle.base_rate, 4),
                model_source=bundle.source,
                trained_at=bundle.trained_at,
                customer_id=payload.get("customer_id"),
            ),
        )

    def explain(self, payload: dict[str, Any]) -> Explanation:
        bundle = self.bundle
        attributions, base = bundle.shap_values(payload)
        feats = bundle.feature_names
        items = [
            FeatureAttribution.of(f, float(payload.get(f, 0.0) or 0.0), float(v))
            for f, v in zip(feats, attributions)
        ]
        items.sort(key=lambda a: abs(a.attribution), reverse=True)

        top = items[0] if items else None
        summary = (
            f"'{top.feature}' is the strongest driver "
            f"({'raising' if top.direction == 'increases_risk' else 'lowering'} churn risk)."
            if top is not None
            else "No attributions available."
        )
        return Explanation(
            kind="shap",
            space="log_odds",
            attributions=items,
            base_value=round(base, 6) if base == base else None,  # NaN guard
            summary=summary,
            caveats=[
                "TreeSHAP on gradient boosting is exact in raw log-odds: base_value + the "
                "attributions reproduce the uncalibrated model's margin. They do NOT sum to the "
                "calibrated probability shown as `score`.",
                "Attributions explain this one record. Global feature importance is a different quantity.",
            ],
        )

    # --- agent tools -----------------------------------------------------------

    def tools(self) -> list[AgentTool]:
        return [
            AgentTool(
                name="score_customer_churn",
                description=(
                    "Score one telecom customer's probability of churning and get a recommended "
                    "retention action. Requires all 13 usage/plan features."
                ),
                module_id=self.id,
                parameters_schema={
                    "type": "object",
                    "properties": {f: {"type": "number", "description": f} for f in FEATURES},
                    "required": list(FEATURES),
                },
            ),
            AgentTool(
                name="rank_at_risk_customers",
                description=(
                    "Rank the loaded customer base by churn risk and return the top N with their "
                    "primary driver. Use this to build a retention campaign target list."
                ),
                module_id=self.id,
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "n": {"type": "integer", "description": "How many customers to return.", "default": 10},
                        "min_score": {"type": "number", "description": "Only return customers above this risk.", "default": 0.0},
                    },
                },
            ),
            AgentTool(
                name="get_churn_model_metrics",
                description=(
                    "Return the churn model's held-out metrics, operating threshold and the "
                    "methodological notes behind them. Use before trusting any prediction."
                ),
                module_id=self.id,
                parameters_schema={"type": "object", "properties": {}},
            ),
        ]

    def tool_score_customer_churn(self, **features: Any) -> dict[str, Any]:
        return self.predict(features).model_dump()

    def tool_rank_at_risk_customers(self, n: int = 10, min_score: float = 0.0) -> dict[str, Any]:
        bundle = self.bundle
        df, source = load_telco()
        X = df[list(FEATURES)]
        proba = bundle.model.predict_proba(X)[:, 1]
        order = proba.argsort()[::-1]

        out = []
        for idx in order:
            score = float(proba[idx])
            if score < min_score:
                continue
            row = df.iloc[int(idx)]
            rec = {f: float(row[f]) for f in FEATURES}
            pred = self.predict(rec)
            out.append(
                {
                    "row_index": int(idx),
                    "customer_id": str(row.get("customer_id", "")),
                    "state": str(row.get("state", "")),
                    "score": round(score, 4),
                    "label": pred.label,
                    "top_driver": pred.factors[0].feature if pred.factors else None,
                    "decision": pred.decision,
                }
            )
            if len(out) >= n:
                break

        return {
            "n_returned": len(out),
            "n_scored": int(len(df)),
            "data_source": source,
            "tenant": "Northwind Global",
            "business_unit": "Northwind Connect",
            "dataset": "Subscriber billing & behaviour extract (IBM Telco Customer Churn, 3,333 rows)",
            "join_key": "customer_id",
            "threshold": bundle.threshold,
            "customers": out,
        }

    def tool_get_churn_model_metrics(self) -> dict[str, Any]:
        report = self.evaluate()
        return report.model_dump()

    # --- helpers ---------------------------------------------------------------

    def _safe_explain(self, payload: dict[str, Any]) -> Explanation | None:
        """Explanations are best-effort; scoring is not. Never let XAI break a prediction."""
        try:
            return self.explain(payload)
        except Exception:  # noqa: BLE001
            return None


#: The singleton the registry discovers.
MODULE = ChurnModule()
