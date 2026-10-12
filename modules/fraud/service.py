"""Fraud & Anomaly — a Saros body.

Also the first body to implement ``stream()``: the contract's optional live-feed capability.
Transactions are generated, scored as they arrive, and yielded — which is the same shape a
Redis Streams or CDC consumer would have, so swapping the source does not touch the body.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any, AsyncIterator

from core.contract import Module
from core.types import AgentTool, Explanation, FeatureAttribution, MetricsReport, ModuleManifest, Prediction, UploadAnalysis
from core.upload_utils import (
    classification_upload_result,
    coerce_binary_column,
    coerce_numeric_columns,
    prediction_inputs,
    read_csv_table,
)

import pandas as pd

from .data import FEATURES, TARGET, synthesize
from .train import FraudModel, train_fraud
from core.explain import build_explainer
import numpy as np

ACCENT = "#f43f5e"


def _verdict(score: float, threshold: float) -> tuple[str, str]:
    if score >= max(threshold + 0.25, 0.9):
        return "Critical fraud risk", "Block the transaction and freeze the card pending customer contact."
    if score >= threshold:
        return "High fraud risk", "Hold for manual review (15-minute SLA). Do not decline outright — a false decline costs churn."
    if score >= threshold * 0.6:
        return "Elevated fraud risk", "Settle, but add the card to the watchlist for the next 24 hours."
    return "Low fraud risk", "Settle normally."


class FraudModule(Module):
    id = "fraud"

    def __init__(self) -> None:
        self._bundle: FraudModel | None = None
        self._report: MetricsReport | None = None
        self._lock = threading.Lock()

    @property
    def bundle(self) -> FraudModel:
        if self._bundle is None:
            with self._lock:
                if self._bundle is None:
                    try:
                        self._bundle = FraudModel.load()
                    except Exception:  # noqa: BLE001 - no artifact yet
                        self._bundle, self._report = train_fraud()
                        self._bundle.save()
        return self._bundle

    def train(self, config: dict[str, Any] | None = None) -> MetricsReport:
        with self._lock:
            bundle, report = train_fraud(**(config or {}))
            bundle.save()
            self._bundle, self._report = bundle, report
        return report

    def evaluate(self) -> MetricsReport:
        if self._report is None:
            # Let any lifespan warm-up finish before falling back to an evaluation train.
            _ = self.bundle
        if self._report is None:
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
            raise ValueError("Fraud analysis requires a CSV upload.")
        frame = read_csv_table(
            content,
            filename,
            module_id=self.id,
            expected_columns=[*FEATURES, TARGET],
            optional_columns=[TARGET],
        )
        frame = coerce_numeric_columns(frame, FEATURES)
        if frame[list(FEATURES)].isna().any().any():
            missing = [column for column in FEATURES if frame[column].isna().any()]
            raise ValueError(f"Fraud feature columns cannot contain blank values: {', '.join(missing)}.")
        if TARGET in frame:
            frame = coerce_binary_column(frame, TARGET)
            if frame[TARGET].isna().any():
                raise ValueError(f"Target column '{TARGET}' cannot contain blank values.")

        bundle = self.bundle
        X = frame[list(FEATURES)]
        supervised = bundle.supervised.predict_proba(X)[:, 1]
        unsupervised = bundle.unsupervised_proba(X)
        probabilities = np.maximum(supervised, unsupervised)
        predictions: list[Prediction] = []
        for index, (_, row) in enumerate(frame.iterrows()):
            score = float(probabilities[index])
            label, decision = _verdict(score, bundle.threshold)
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
                        "supervised_score": round(float(supervised[index]), 4),
                        "unsupervised_score": round(float(unsupervised[index]), 4),
                        "decisive_model": "supervised model" if supervised[index] >= unsupervised[index] else "isolation forest",
                        "base_rate": round(bundle.base_rate, 5),
                        "training_rows": bundle.n_train,
                        "training_data_source": "synthetic deterministic simulation; not real company transactions",
                        "model_basis": "max(supervised gradient-boosted score, calibrated isolation-forest anomaly score)",
                        "model_features": list(FEATURES),
                        "uploaded_data": True,
                    },
                )
            )
        explained = 0
        for prediction in sorted(predictions, key=lambda item: item.score, reverse=True)[:3]:
            if prediction.meta.get("decisive_model") == "supervised model":
                try:
                    explanation = self.explain(prediction.inputs)
                    prediction.explanation = explanation
                    prediction.factors = explanation.attributions[:5]
                    explained += int(bool(explanation.attributions))
                except Exception:  # noqa: BLE001 - preserve scores if optional XAI fails
                    prediction.explanation = Explanation(
                        kind="none",
                        summary="The supervised score was decisive, but local feature attributions were unavailable.",
                    )
            else:
                prediction.explanation = Explanation(
                    kind="none",
                    summary=(
                        "The calibrated isolation-forest anomaly score was higher than the supervised score. "
                        "This pipeline does not provide feature-level attributions for the forest."
                    ),
                    caveats=[
                        "Both component scores and the decisive model are included in the prediction metadata.",
                    ],
                )

        labels = frame[TARGET].astype(int).tolist() if TARGET in frame else None
        analysis = classification_upload_result(
            module_id=self.id,
            filename=filename,
            n_rows=len(frame),
            predictions=predictions,
            threshold=bundle.threshold,
            target_name=TARGET,
            labels=labels,
            model_source="supervised and anomaly-detection ensemble",
        )
        analysis.metrics.notes.insert(
            0,
            "Training provenance: the built-in fraud model uses a deterministic synthetic "
            "demonstration transaction stream, not transactions from a real company. "
            "The uploaded CSV was scored with that existing model and was not used to train it.",
        )
        analysis.metrics.notes.append(
            f"For the three highest-scored returned rows, supervised-model SHAP was available for "
            f"{explained} row(s); isolation-forest decisions are accompanied by component scores "
            "but do not have feature-level attribution in this pipeline."
        )
        return analysis

    # --- contract ------------------------------------------------------------

    @property
    def manifest(self) -> ModuleManifest:
        return ModuleManifest(
            id=self.id,
            title="Fraud & Anomaly",
            summary=(
                "Transaction scoring with a supervised gradient-boosted model plus an isolation "
                "forest; the higher risk score wins. The built-in training and live-feed examples "
                "are synthetic demonstrations, not data from a real company. Upload a CSV to score "
                "your own transactions; uploads do not retrain the model."
            ),
            accent=ACCENT,
            status="live",
            capabilities=["classification", "anomaly-detection", "imbalance-handling", "streaming", "shap-xai"],
            routes={"ui": "/fraud", "api": "/api/v1/fraud"},
            tools=self.tools(),
        )

    def predict(self, payload: dict[str, Any]) -> Prediction:
        bundle = self.bundle
        combined, sup, unsup = bundle.predict_proba(payload)
        label, decision = _verdict(combined, bundle.threshold)

        # Whichever model fired is the honest "why" — do not blend the two into one story.
        which = "supervised model" if sup >= unsup else "isolation forest"
        return Prediction(
            module_id=self.id,
            score=round(float(combined), 4),
            label=label,
            decision=decision,
            inputs={k: float(payload[k]) for k in FEATURES if payload.get(k) is not None},
            factors=[],
            meta={
                "threshold": bundle.threshold,
                "supervised_score": round(float(sup), 4),
                "unsupervised_score": round(float(unsup), 4),
                "decisive_model": which,
                "base_rate": round(bundle.base_rate, 5),
                "training_rows": bundle.n_train,
                "training_data_source": "synthetic deterministic simulation; not real company transactions",
                "model_basis": "max(supervised gradient-boosted score, calibrated isolation-forest anomaly score)",
                "model_features": list(FEATURES),
            },
        )

    def explain(self, payload: dict[str, Any]) -> Explanation:
        import shap

        bundle = self.bundle
        X = bundle.frame(payload)
        try:
            explainer = build_explainer(bundle.supervised).explainer
            sv = explainer.shap_values(X)
            sv = sv[:, :, -1] if getattr(sv, "ndim", 0) == 3 else sv
            import numpy as _np

            base = float(_np.ravel(getattr(explainer, "expected_value", 0.0))[-1])
        except Exception as exc:  # noqa: BLE001 - explanation must never break scoring
            return Explanation(kind="none", summary=f"Explanation unavailable: {exc}")

        items = [
            FeatureAttribution.of(f, float(payload.get(f, 0.0) or 0.0), float(v))
            for f, v in zip(bundle.feature_names, sv[0])
        ]
        items.sort(key=lambda a: abs(a.attribution), reverse=True)
        top = items[0] if items else None

        return Explanation(
            kind="shap",
            space="log_odds",
            attributions=items,
            base_value=round(base, 6),
            summary=(
                f"'{top.feature}' is the strongest driver "
                f"({'raising' if top.direction == 'increases_risk' else 'lowering'} fraud risk)."
                if top else "No attributions available."
            ),
            caveats=[
                "These attributions explain the *supervised* model only. If the isolation forest "
                "produced the final score, it fired on a pattern with no feature-level story — "
                "which is exactly what it is for.",
                "TreeSHAP is exact in log-odds and does not sum to the displayed probability.",
            ],
        )

    # --- streaming -----------------------------------------------------------

    async def stream(self) -> AsyncIterator[dict[str, Any]]:
        """Yield scored transactions as they 'arrive'.

        Simulated arrival with jitter. A real deployment replaces this loop with a Redis
        Streams consumer; nothing else in the body changes.
        """
        bundle = self.bundle
        # Fraud is ~1% of traffic, so an unbiased 60-transaction window usually contains no
        # fraud at all and the live feed looks dead. Enrich it for the demo — and say so,
        # rather than quietly implying a real feed flags this often.
        df = synthesize(n=2000, seed=99)
        fraud = df[df["is_fraud"] == 1].head(12)
        legit = df[df["is_fraud"] == 0].head(48)
        df = pd.concat([fraud, legit]).sample(frac=1.0, random_state=1).reset_index(drop=True)
        for i, (_, row) in enumerate(df.iterrows()):
            payload = {f: float(row[f]) for f in FEATURES}
            combined, sup, unsup = bundle.predict_proba(payload)
            label, _ = _verdict(combined, bundle.threshold)
            yield {
                "seq": i,
                "txn_id": f"txn_{i:05d}",
                "amount": float(row["amount"]),
                "hour_of_day": float(row["hour_of_day"]),
                "risk": round(float(combined), 4),
                "supervised": round(float(sup), 4),
                "unsupervised": round(float(unsup), 4),
                "label": label,
                "flagged": bool(combined >= bundle.threshold),
                "actual_fraud": bool(row["is_fraud"]),
            }
            await asyncio.sleep(0.35)
        yield {
            "seq": -1,
            "note": (
                "This feed is fraud-enriched for demonstration. Real traffic is ~1% "
                "fraudulent, so most windows would show nothing to flag."
            ),
        }

    # --- agent tools ---------------------------------------------------------

    def tools(self) -> list[AgentTool]:
        return [
            AgentTool(
                name="score_transaction",
                description=(
                    "Score a single transaction for fraud. Returns a risk score, the verdict, "
                    "and which model was decisive. Requires amount, velocity and card features."
                ),
                module_id=self.id,
                parameters_schema={
                    "type": "object",
                    "properties": {f: {"type": "number", "description": f} for f in FEATURES},
                    "required": list(FEATURES),
                },
            ),
            AgentTool(
                name="summarise_fraud_stream",
                description=(
                    "Score a batch of recent transactions and summarise the flagged ones. Use "
                    "this for 'what is happening on the fraud feed right now'."
                ),
                module_id=self.id,
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "n": {"type": "integer", "description": "Transactions to sample.", "default": 60},
                    },
                },
            ),
            AgentTool(
                name="check_fraud_drift",
                description=(
                    "Check whether recent transactions still look like the data the fraud model "
                    "was trained on. Reports per-feature PSI and an overall verdict. Use this for "
                    "'has the fraud feed drifted' or 'is the model still valid'."
                ),
                module_id=self.id,
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "n": {"type": "integer", "description": "Batch size to sample."},
                    },
                },
            ),
            AgentTool(
                name="get_fraud_model_metrics",
                description="Held-out fraud model metrics, threshold, and methodology notes.",
                module_id=self.id,
                parameters_schema={"type": "object", "properties": {}},
            ),
        ]

    def tool_score_transaction(self, **features: Any) -> dict[str, Any]:
        return self.predict(features).model_dump()

    def tool_summarise_fraud_stream(self, n: int = 60) -> dict[str, Any]:
        bundle = self.bundle
        pool = synthesize(n=4000, seed=99)
        # Same enrichment as the live feed, for the same reason, stated in the output.
        fraud = pool[pool["is_fraud"] == 1].head(max(4, n // 8))
        legit = pool[pool["is_fraud"] == 0].head(min(n, 400))
        df = pd.concat([fraud, legit]).sample(frac=1.0, random_state=2).reset_index(drop=True)
        rows = []
        for _, row in df.iterrows():
            payload = {f: float(row[f]) for f in FEATURES}
            combined, sup, unsup = bundle.predict_proba(payload)
            rows.append(
                {
                    "amount": float(row["amount"]),
                    "risk": round(float(combined), 4),
                    "flagged": bool(combined >= bundle.threshold),
                    "actual_fraud": bool(row["is_fraud"]),
                    "decisive": "supervised" if sup >= unsup else "isolation_forest",
                }
            )
        flagged = [r for r in rows if r["flagged"]]
        caught = sum(1 for r in rows if r["flagged"] and r["actual_fraud"])
        missed = sum(1 for r in rows if not r["flagged"] and r["actual_fraud"])
        false_alarms = sum(1 for r in rows if r["flagged"] and not r["actual_fraud"])
        return {
            "n_scored": len(rows),
            "n_flagged": len(flagged),
            "true_positives": caught,
            "false_positives": false_alarms,
            "missed_fraud": missed,
            "threshold": bundle.threshold,
            "enriched": True,
            "data_source": "synthetic fraud-enriched demonstration stream; not real company transactions",
            "enrichment_note": (
                "Synthetic demonstration sample is fraud-enriched so the summary is informative; "
                "it is not a live company feed, and real traffic is ~1% fraudulent."
            ),
            "top_flagged": sorted(flagged, key=lambda r: -r["risk"])[:5],
        }

    def tool_check_fraud_drift(self, n: int = 400) -> dict[str, Any]:
        """Compare a fresh batch of transactions against the distribution the model trained on.

        A model that stays confident while its input distribution moves is the failure mode that
        matters for a real-time fraud feed: nothing errors, the scores keep coming, and they
        stop meaning what they meant at training time. PSI per feature is how that becomes
        visible instead of silent.
        """
        from core.drift import drift_report

        bundle = self.bundle
        if not bundle.reference:
            return {"error": "No training distribution retained; cannot assess drift."}

        batch = synthesize(n=max(50, int(n)), seed=4242)
        actual = {f: np.asarray(batch[f].values, dtype=float)
                  for f in bundle.feature_names if f in batch.columns}

        out = drift_report(bundle.reference, actual)
        out["batch_size"] = len(batch)
        out["reference_size"] = bundle.n_train
        out["note"] = (
            "Compared against the fit set the supervised model was trained on. Bin edges are cut "
            "on that reference and never move, which is what makes the comparison meaningful."
        )
        return out

    def tool_get_fraud_model_metrics(self) -> dict[str, Any]:
        return self.evaluate().model_dump()


MODULE = FraudModule()
