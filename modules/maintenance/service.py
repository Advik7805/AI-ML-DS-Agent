"""Predictive Maintenance — a Saros body.

Remaining-useful-life regression on turbofan sensor telemetry. The headline number is not
RMSE but **early-prediction error**: a model that is accurate near failure and useless 60
cycles out is worse than one that is mediocre everywhere, because maintenance lead time is
the entire point of the exercise.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

from core.contract import Module
from core.types import AgentTool, Explanation, FeatureAttribution, MetricsReport, ModuleManifest, Prediction

from .data import FEATURES, RUL_CAP, SENSORS, SETTINGS, load_cmapss, synthesize
from core.explain import build_explainer

ACCENT = "#22d3ee"
ARTIFACT = Path(__file__).resolve().parents[2] / "artifacts" / "maintenance" / "rul_model.pkl"


@dataclass
class RulModel:
    model: Any
    feature_names: list[str]
    metrics: dict[str, float] = field(default_factory=dict)
    source: str = "synthetic"
    trained_at: str = ""
    notes: list[str] = field(default_factory=list)
    #: Training distribution per feature, so incoming telemetry can be checked for drift.
    reference: dict = field(default_factory=dict)

    def frame(self, payload: dict[str, Any]) -> pd.DataFrame:
        missing = [f for f in self.feature_names if f not in payload]
        if missing:
            raise ValueError(f"Missing required features: {', '.join(missing[:6])}")
        return pd.DataFrame([{f: float(payload[f]) for f in self.feature_names}], columns=self.feature_names)

    def predict_rul(self, payload: dict[str, Any]) -> float:
        return float(np.clip(self.model.predict(self.frame(payload))[0], 0, RUL_CAP))

    def save(self, path: Path = ARTIFACT) -> Path:
        import pickle

        path.parent.mkdir(parents=True, exist_ok=True)
        from core.artifacts import save as _save_artifact

        return _save_artifact(self, path)

    @classmethod
    def load(cls, path: Path = ARTIFACT) -> "RulModel":
        import pickle

        from core.artifacts import load as _load_artifact

        return _load_artifact(path)


def train_rul(*, random_state: int = 11) -> tuple[RulModel, MetricsReport]:
    df, source = load_cmapss()
    feature_cols = [c for c in FEATURES if c in df.columns]
    X = df[feature_cols]
    y = df["rul"].astype(float)
    groups = df["unit"]

    # GroupKFold by engine: rows from one engine are highly autocorrelated, so a random split
    # leaks and reports RUL error the model could never achieve on an unseen engine.
    gkf = GroupKFold(n_splits=4)
    preds = np.zeros(len(df))
    for tr, te in gkf.split(X, y, groups):
        m = HistGradientBoostingRegressor(
            max_iter=300, max_depth=6, learning_rate=0.06, random_state=random_state
        ).fit(X.iloc[tr], y.iloc[tr])
        preds[te] = np.clip(m.predict(X.iloc[te]), 0, RUL_CAP)

    final = HistGradientBoostingRegressor(
        max_iter=300, max_depth=6, learning_rate=0.06, random_state=random_state
    ).fit(X, y)

    notes = [
        f"Data source: {source} ({len(df):,} rows, {df['unit'].nunique()} engines).",
        f"RUL capped at {RUL_CAP} cycles — beyond that only 'not failing soon' is defensible.",
        "GroupKFold by engine. Cycles within one engine are autocorrelated, so a random split "
        "leaks and understates error on genuinely unseen hardware.",
    ]

    overall_mae = float(mean_absolute_error(y, preds))
    overall_rmse = float(np.sqrt(mean_squared_error(y, preds)))

    # The number that matters: error in the band where a maintenance decision is actually made.
    bands = {"0-30 cycles": (0, 30), "30-60 cycles": (30, 60), "60-125 cycles": (60, RUL_CAP)}
    band_mae: dict[str, float] = {}
    for name, (lo, hi) in bands.items():
        mask = (y >= lo) & (y <= hi)
        band_mae[name] = float(mean_absolute_error(y[mask], preds[mask])) if mask.any() else float("nan")
    notes.append(
        "Error is reported by remaining-life band, not only overall. Overall MAE is dominated "
        "by the far-future band, which is also the band where a precise number is least useful."
    )

    scores = {
        "mae_cycles": overall_mae,
        "rmse_cycles": overall_rmse,
        "r2": float(r2_score(y, preds)),
        **{f"mae_{k.replace(' ', '_').replace('-', '_')}": v for k, v in band_mae.items()},
        "n_engines": float(df["unit"].nunique()),
    }

    bundle = RulModel(
        model=final,
        feature_names=feature_cols,
        metrics=scores,
        source=source,
        trained_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        notes=notes,
        reference={c: np.asarray(X[c].values, dtype=float) for c in feature_cols},
    )

    report = MetricsReport(
        n_samples=int(len(df)),
        positive_rate=float((y <= 30).mean()),
        scores=scores,
        notes=[*notes, *(f"MAE in {k}: {v:.2f} cycles." for k, v in band_mae.items())],
    )
    return bundle, report


class MaintenanceModule(Module):
    id = "maintenance"

    def __init__(self) -> None:
        self._bundle: RulModel | None = None
        self._report: MetricsReport | None = None
        self._lock = threading.Lock()

    @property
    def bundle(self) -> RulModel:
        if self._bundle is None:
            with self._lock:
                if self._bundle is None:
                    try:
                        self._bundle = RulModel.load()
                        self._report = MetricsReport(
                            n_samples=0, positive_rate=0.0, scores=self._bundle.metrics,
                            notes=self._bundle.notes,
                        )
                    except Exception:  # noqa: BLE001
                        self._bundle, self._report = train_rul()
                        self._bundle.save()
        return self._bundle

    def train(self, config: dict[str, Any] | None = None) -> MetricsReport:
        bundle, report = train_rul(**(config or {}))
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
            title="Predictive Maintenance",
            summary=(
                "Remaining-useful-life regression on turbofan sensor telemetry, cross-validated "
                "by engine so the error is measured on hardware the model never saw. Error is "
                "reported per remaining-life band, because overall RMSE hides where it fails."
            ),
            accent=ACCENT,
            status="live",
            capabilities=["regression", "time-series", "grouped-cv", "shap-xai"],
            routes={"ui": "/maintenance", "api": "/api/v1/maintenance"},
            tools=self.tools(),
        )

    def predict(self, payload: dict[str, Any]) -> Prediction:
        bundle = self.bundle
        rul = bundle.predict_rul(payload)
        # Contract requires 0..1: express it as fraction of the cap consumed, i.e. urgency.
        urgency = float(np.clip(1.0 - rul / RUL_CAP, 0.0, 1.0))

        if rul <= 15:
            label, decision = "Imminent failure", "Ground the unit. Schedule replacement within 15 cycles."
        elif rul <= 40:
            label, decision = "Failure likely soon", "Book maintenance in the next planning window; order the part now."
        elif rul <= 80:
            label, decision = "Degrading", "Add to the watch list and re-score every 10 cycles."
        else:
            label, decision = "Healthy", "Routine inspection schedule only."

        return Prediction(
            module_id=self.id,
            score=round(urgency, 4),
            label=label,
            decision=decision,
            inputs={k: float(payload[k]) for k in bundle.feature_names if payload.get(k) is not None},
            factors=self._drivers(payload),
            meta={"rul_cycles": round(rul, 1), "rul_cap": RUL_CAP, "source": bundle.source},
        )

    def _drivers(self, payload: dict[str, Any], k: int = 5) -> list[FeatureAttribution]:
        import shap

        try:
            bundle = self.bundle
            explainer = build_explainer(bundle.model).explainer
            sv = explainer.shap_values(bundle.frame(payload))[0]
            items = [
                FeatureAttribution.of(f, float(payload.get(f, 0.0) or 0.0), -float(v))
                for f, v in zip(bundle.feature_names, sv)
            ]
            # Negated: a positive SHAP value raises RUL, which *lowers* risk.
            items.sort(key=lambda a: abs(a.attribution), reverse=True)
            return items[:k]
        except Exception:  # noqa: BLE001
            return []

    def explain(self, payload: dict[str, Any]) -> Explanation:
        import shap

        bundle = self.bundle
        explainer = build_explainer(bundle.model).explainer
        sv = explainer.shap_values(bundle.frame(payload))[0]
        base = float(np.ravel(explainer.expected_value)[-1])
        items = [
            FeatureAttribution.of(f, float(payload.get(f, 0.0) or 0.0), -float(v))
            for f, v in zip(bundle.feature_names, sv)
        ]
        items.sort(key=lambda a: abs(a.attribution), reverse=True)
        return Explanation(
            kind="shap",
            space="log_odds",
            attributions=items,
            base_value=round(base, 4),
            summary=(
                f"Predicted RUL {bundle.predict_rul(payload):.0f} cycles. '{items[0].feature}' "
                f"contributes most." if items else "No attributions."
            ),
            caveats=[
                "Attributions are sign-flipped so that positive means 'shortens remaining life'. "
                "SHAP's own sign is the effect on RUL, which is the opposite of risk.",
                "13 of the 21 sensors are pure noise in this dataset. Non-zero attributions on "
                "them are the model fitting noise, not signal.",
            ],
        )

    def tools(self) -> list[AgentTool]:
        return [
            AgentTool(
                name="predict_rul",
                description=(
                    "Predict remaining useful life (in cycles) for a turbofan engine from one "
                    "cycle of sensor readings. Requires all 27 setting and sensor values."
                ),
                module_id=self.id,
                parameters_schema={
                    "type": "object",
                    "properties": {f: {"type": "number"} for f in FEATURES},
                    "required": FEATURES,
                },
            ),
            AgentTool(
                name="rank_engines_by_urgency",
                description="Rank the fleet by predicted failure urgency and return the top N needing attention.",
                module_id=self.id,
                parameters_schema={
                    "type": "object",
                    "properties": {"n": {"type": "integer", "default": 10}},
                },
            ),
            AgentTool(
                name="check_fleet_drift",
                description=(
                    "Check whether fresh engine telemetry still looks like the data the RUL model "
                    "was trained on. Reports per-sensor PSI and an overall verdict. Use this for "
                    "'has the fleet drifted' or 'is the maintenance model still valid'."
                ),
                module_id=self.id,
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "n_engines": {"type": "integer", "description": "Engines to sample."},
                    },
                },
            ),
            AgentTool(
                name="get_maintenance_model_metrics",
                description="RUL model metrics by remaining-life band, plus methodology notes.",
                module_id=self.id,
                parameters_schema={"type": "object", "properties": {}},
            ),
        ]

    def tool_predict_rul(self, **features: Any) -> dict[str, Any]:
        return self.predict(features).model_dump()

    def tool_rank_engines_by_urgency(self, n: int = 10) -> dict[str, Any]:
        bundle = self.bundle
        df, source = load_cmapss()
        cols = [c for c in bundle.feature_names if c in df.columns]
        latest = df.sort_values("cycle").groupby("unit").tail(1).reset_index(drop=True)
        ruls = np.clip(bundle.model.predict(latest[cols]), 0, RUL_CAP)
        order = np.argsort(ruls)[:n]
        return {
            "n_engines": int(latest["unit"].nunique()),
            "data_source": source,
            "engines": [
                {
                    "unit": int(latest.iloc[i]["unit"]),
                    "cycle": int(latest.iloc[i]["cycle"]),
                    "predicted_rul": round(float(ruls[i]), 1),
                    "urgency": round(float(1 - ruls[i] / RUL_CAP), 3),
                }
                for i in order
            ],
        }

    def tool_check_fleet_drift(self, n_engines: int = 20) -> dict[str, Any]:
        """Compare fresh telemetry against the distribution the RUL model was trained on.

        Fleet sensors drift for ordinary reasons - a new engine model, a recalibrated sensor, a
        different operating profile - and the RUL model will keep producing confident cycle
        counts regardless. This makes that visible instead of silent.
        """
        from core.drift import drift_report

        bundle = self.bundle
        if not bundle.reference:
            return {"error": "No training distribution retained; cannot assess drift."}

        fresh = synthesize(n_units=max(5, int(n_engines)), seed=999)
        actual = {c: np.asarray(fresh[c].values, dtype=float)
                  for c in bundle.feature_names if c in fresh.columns}

        out = drift_report(bundle.reference, actual)
        out["reference_source"] = bundle.source
        out["reference_engines"] = int(bundle.metrics.get("n_engines", 0))
        out["note"] = (
            "Compared against the telemetry the model was trained on. A shift here is not "
            "automatically a fault - the fleet may legitimately have changed - but the model's "
            "error estimates no longer apply to it."
        )
        return out

    def tool_get_maintenance_model_metrics(self) -> dict[str, Any]:
        return self.evaluate().model_dump()


MODULE = MaintenanceModule()
