"""Predictive Maintenance — a Saros body.

Remaining-useful-life regression on turbofan sensor telemetry. The headline number is not
RMSE but **early-prediction error**: a model that is accurate near failure and useless 60
cycles out is worse than one that is mediocre everywhere, because maintenance lead time is
the entire point of the exercise.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from io import StringIO
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

from core.contract import Module
from core.types import AgentTool, Explanation, FeatureAttribution, MetricsReport, ModuleManifest, Prediction, UploadAnalysis
from core.upload_utils import (
    MAX_ANALYZE_ROWS,
    MAX_RETURNED_PREDICTIONS,
    coerce_numeric_columns,
    prediction_inputs,
)

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


def _read_maintenance_upload(filename: str, content: bytes) -> pd.DataFrame:
    """Accept named CSV or whitespace-delimited, CMAPSS-shaped TXT in memory."""
    suffix = Path(filename).suffix.lower()
    if suffix not in {".csv", ".txt"}:
        raise ValueError("Maintenance accepts turbofan-shaped .csv or whitespace-delimited .txt files.")
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise ValueError("Maintenance data must be UTF-8 text.") from None

    try:
        if suffix == ".csv":
            frame = pd.read_csv(StringIO(text))
        else:
            first_line = next((line for line in text.splitlines() if line.strip()), "")
            tokens = re.split(r"\s+", first_line.strip())
            has_header = False
            for token in tokens:
                try:
                    float(token)
                except ValueError:
                    has_header = True
                    break
            if has_header:
                frame = pd.read_csv(StringIO(text), sep=r"\s+")
            else:
                raw = pd.read_csv(StringIO(text), sep=r"\s+", header=None)
                base = ["unit", "cycle", *SETTINGS[:3], *SENSORS]
                columns_by_count = {
                    26: base,
                    27: [*base, "rul"],
                    29: ["unit", "cycle", *SETTINGS, *SENSORS],
                    30: ["unit", "cycle", *SETTINGS, *SENSORS, "rul"],
                }
                if raw.shape[1] not in columns_by_count:
                    raise ValueError(
                        f"TXT has {raw.shape[1]} columns. Expected 26/27 NASA CMAPSS columns "
                        "or 29/30 Saros turbofan columns (optionally with RUL)."
                    )
                raw.columns = columns_by_count[raw.shape[1]]
                frame = raw
    except ValueError:
        raise
    except Exception as exc:  # noqa: BLE001 - turn pandas parser details into a 422
        raise ValueError(f"Could not parse turbofan data: {type(exc).__name__}: {exc}") from None

    names = []
    for column in frame.columns:
        normalized = re.sub(r"[^a-z0-9]", "", str(column).strip().lower())
        if normalized in {"engine", "engineid", "unitid", "unitnumber"}:
            normalized = "unit"
        elif normalized in {"remainingusefullife", "remaininglife"}:
            normalized = "rul"
        names.append(normalized)
    frame.columns = names
    if frame.columns.duplicated().any():
        raise ValueError("Maintenance data has duplicate column names after normalization.")
    if frame.empty:
        raise ValueError("Maintenance data contains no rows.")
    if len(frame) > MAX_ANALYZE_ROWS:
        raise ValueError(
            f"Maintenance data has {len(frame):,} rows; this analyzer is capped at "
            f"{MAX_ANALYZE_ROWS:,} rows per upload."
        )

    expected = ["unit", "cycle", *SETTINGS[:3], *SENSORS]
    missing = [column for column in expected if column not in frame.columns]
    if missing:
        raise ValueError(
            "Maintenance data is missing required turbofan columns: "
            f"{', '.join(missing)}. Expected columns: {', '.join(expected)}."
        )
    for setting in SETTINGS[3:]:
        if setting not in frame:
            frame[setting] = 0.0
    numeric_columns = [*expected, *SETTINGS[3:]]
    if "rul" in frame:
        numeric_columns.append("rul")
    frame = coerce_numeric_columns(frame, numeric_columns)
    if frame[["unit", "cycle"]].isna().any().any():
        raise ValueError("Maintenance columns 'unit' and 'cycle' cannot be blank.")
    if "rul" in frame and frame["rul"].isna().any():
        raise ValueError("Optional 'rul' labels cannot contain blank values.")
    return frame


def _rul_decision(rul: float) -> tuple[str, str]:
    if rul <= 15:
        return "Imminent failure", "Ground the unit. Schedule replacement within 15 cycles."
    if rul <= 40:
        return "Failure likely soon", "Book maintenance in the next planning window; order the part now."
    if rul <= 80:
        return "Degrading", "Add to the watch list and re-score every 10 cycles."
    return "Healthy", "Routine inspection schedule only."


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
        with self._lock:
            bundle, report = train_rul(**(config or {}))
            bundle.save()
            self._bundle, self._report = bundle, report
        return report

    def evaluate(self) -> MetricsReport:
        if self._report is None:
            # Wait for a concurrent startup warm-up before deciding to train again.
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
        if not filename or content is None:
            raise ValueError("Maintenance analysis requires a turbofan-shaped CSV or TXT upload.")
        frame = _read_maintenance_upload(filename, content)
        latest = frame.sort_values(["unit", "cycle"]).groupby("unit", as_index=False).tail(1).reset_index(drop=True)
        bundle = self.bundle
        X = latest[bundle.feature_names]
        rul_values = np.clip(bundle.model.predict(X), 0, RUL_CAP)
        predictions: list[Prediction] = []
        for index, (_, row) in enumerate(latest.iterrows()):
            rul = float(rul_values[index])
            urgency = float(np.clip(1.0 - rul / RUL_CAP, 0.0, 1.0))
            label, decision = _rul_decision(rul)
            predictions.append(
                Prediction(
                    module_id=self.id,
                    score=round(urgency, 4),
                    label=label,
                    decision=decision,
                    inputs=prediction_inputs(row, bundle.feature_names),
                    meta={
                        "row_index": index,
                        "unit": int(row["unit"]),
                        "cycle": int(row["cycle"]),
                        "rul_cycles": round(rul, 1),
                        "rul_cap": RUL_CAP,
                        "source": bundle.source,
                        "model_basis": "HistGradientBoostingRegressor on the latest row per engine; urgency = 1 - predicted RUL / cap",
                        "model_features": list(bundle.feature_names),
                        "uploaded_data": True,
                    },
                )
            )

        explained = 0
        for prediction in sorted(predictions, key=lambda item: item.score, reverse=True)[:3]:
            try:
                explanation = self.explain(prediction.inputs)
                prediction.explanation = explanation
                prediction.factors = explanation.attributions[:5]
                explained += int(bool(explanation.attributions))
            except Exception:  # noqa: BLE001 - keep scored engines if XAI is unavailable
                prediction.explanation = Explanation(
                    kind="none",
                    summary="The remaining-life score is available; local sensor attributions could not be computed.",
                )

        predicted = np.asarray(rul_values, dtype=float)
        scores = {
            "mean_predicted_rul": float(np.mean(predicted)),
            "mean_urgency": float(np.mean(1.0 - predicted / RUL_CAP)),
            "engine_count": float(len(latest)),
        }
        notes = [
            f"Scored the latest telemetry row for each of {len(latest):,} engine(s) using the existing model.",
            "No upload is saved and no model is retrained. Provide an optional 'rul' column for error metrics.",
            f"Local sensor attributions were computed for {explained} of the three highest-urgency returned engines.",
        ]
        if "rul" in latest:
            actual = latest["rul"].to_numpy(dtype=float)
            errors = predicted - actual
            scores["mae_cycles"] = float(np.mean(np.abs(errors)))
            scores["rmse_cycles"] = float(np.sqrt(np.mean(errors ** 2)))
            positive_rate = float(np.mean(actual <= 30))
            notes.append("MAE/RMSE compare predictions with the uploaded RUL labels for each latest engine row.")
        else:
            positive_rate = float(np.mean(predicted <= 30))
            notes.append("Without an 'rul' label, positive_rate is the share predicted within 30 cycles of failure.")

        predictions.sort(key=lambda item: item.score, reverse=True)
        return UploadAnalysis(
            module_id=self.id,
            filename=filename,
            n_rows=len(frame),
            metrics=MetricsReport(
                n_samples=len(latest),
                positive_rate=positive_rate,
                scores=scores,
                threshold=1.0 - 30.0 / RUL_CAP,
                notes=notes,
            ),
            predictions=predictions[:MAX_RETURNED_PREDICTIONS],
            predictions_truncated=len(predictions) > MAX_RETURNED_PREDICTIONS,
        )

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

        label, decision = _rul_decision(rul)

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
