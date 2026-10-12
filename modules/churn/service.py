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
from core.types import AgentTool, Explanation, FeatureAttribution, MetricsReport, ModuleManifest, Prediction, UploadAnalysis
from core.upload_utils import (
    classification_upload_result,
    coerce_binary_column,
    coerce_numeric_columns,
    read_csv_table,
    prediction_inputs,
)

from .data import BINARY_FEATURES, FEATURES, TARGET, load_telco
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
        with self._lock:
            bundle, report = train_churn(**config)
            bundle.save()
            self._bundle, self._report = bundle, report
        return report

    def evaluate(self) -> MetricsReport:
        if self._report is None:
            # Wait for a concurrent startup warm-up before deciding to retrain.
            _ = self.bundle
        if self._report is None:
            # The artifact does not store metrics, so a loaded artifact still needs evaluation.
            self.train()
        assert self._report is not None
        return self._report

    def analyze_upload(
        self,
        filename: str | None,
        content: bytes | None,
        *,
        session_id: str | None = None,
        query: str = "",
    ) -> UploadAnalysis:
        del session_id, query
        if not filename or not filename.lower().endswith(".csv") or content is None:
            raise ValueError("Churn analysis requires a CSV upload.")
        expected = [*FEATURES, TARGET]
        frame = read_csv_table(
            content,
            filename,
            module_id=self.id,
            expected_columns=expected,
            optional_columns=[TARGET],
        )
        for column in BINARY_FEATURES:
            frame = coerce_binary_column(frame, column)
        if TARGET in frame:
            frame = coerce_binary_column(frame, TARGET)
            if frame[TARGET].isna().any():
                raise ValueError(f"Target column '{TARGET}' cannot contain blank values.")
        frame = coerce_numeric_columns(frame, FEATURES)

        bundle = self.bundle
        probabilities = bundle.model.predict_proba(frame[list(FEATURES)])[:, 1]
        predictions: list[Prediction] = []
        for index, (_, row) in enumerate(frame.iterrows()):
            score = float(probabilities[index])
            label, decision = _band(score, bundle.threshold)
            predictions.append(
                Prediction(
                    module_id=self.id,
                    score=round(score, 4),
                    label=label,
                    decision=decision,
                    inputs=prediction_inputs(row, FEATURES),
                    meta={
                        "row_index": index,
                        "threshold": bundle.threshold,
                        "base_rate": round(bundle.base_rate, 4),
                        "model_source": bundle.source,
                        "uploaded_data": True,
                    },
                )
            )
        labels = frame[TARGET].astype(int).tolist() if TARGET in frame else None
        return classification_upload_result(
            module_id=self.id,
            filename=filename,
            n_rows=len(frame),
            predictions=predictions,
            threshold=bundle.threshold,
            target_name=TARGET,
            labels=labels,
            model_source=bundle.source,
        )

    # --- contract --------------------------------------------------------------

    @property
    def manifest(self) -> ModuleManifest:
        return ModuleManifest(
            id=self.id,
            title="Churn & Retention",
            summary=(
                "Gradient-boosted churn scoring with SMOTE for class imbalance, isotonic "
                "calibration, and SHAP explanations. Ranks an entire book of business by "
                "attrition risk and recommends a retention action per customer."
            ),
            accent=ACCENT,
            status="live",
            capabilities=["classification", "imbalance-handling", "calibration", "shap-xai", "ranking"],
            routes={"ui": "/churn", "api": "/api/v1/churn"},
            tools=self.tools(),
        )

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
            meta={
                "threshold": bundle.threshold,
                "base_rate": round(bundle.base_rate, 4),
                "model_source": bundle.source,
                "trained_at": bundle.trained_at,
            },
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
