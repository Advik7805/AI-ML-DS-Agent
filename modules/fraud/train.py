"""Fraud module — training.

Two models, deliberately:

- a **supervised** gradient-boosted classifier on labelled history, and
- an **unsupervised** isolation forest that catches novel patterns the labels do not cover.

Production fraud needs both: supervised models are precise about what they have seen and blind
to what they have not. The final score is the max of the two, so a novel pattern still trips
the alarm even with no precedent in the training data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from core.types import MetricsReport

from .data import FEATURES, TARGET, synthesize


@dataclass
class FraudModel:
    supervised: Any
    unsupervised: Any
    feature_names: list[str]
    #: Platt map from the isolation forest's decision_function to a real probability.
    unsup_calibrator: Any = None
    threshold: float = 0.5
    metrics: dict[str, float] = field(default_factory=dict)
    trained_at: str = ""
    n_train: int = 0
    base_rate: float = 0.0
    notes: list[str] = field(default_factory=list)

    def frame(self, payload: dict[str, Any]) -> pd.DataFrame:
        missing = [f for f in self.feature_names if f not in payload]
        if missing:
            raise ValueError(f"Missing required features: {', '.join(missing)}")
        row = {f: float(payload[f]) if payload[f] is not None else np.nan for f in self.feature_names}
        return pd.DataFrame([row], columns=self.feature_names)

    def predict_proba(self, payload: dict[str, Any]) -> tuple[float, float, float]:
        """Return (combined, supervised, unsupervised) risk scores."""
        X = self.frame(payload)
        sup = float(self.supervised.predict_proba(X)[0, 1])
        unsup = self.unsupervised_proba(X)[0]
        return max(sup, unsup), sup, unsup

    def unsupervised_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Calibrated anomaly probability.

        The naive ``1/(1+exp(score*6))`` this replaced scored a completely ordinary
        transaction at 0.94, because it never accounted for where the *normal* population
        sits. Platt scaling on a held-out fold does: it learns the offset, so an ordinary
        transaction lands near 0 and the score means something.
        """
        raw = self.unsupervised.decision_function(X).reshape(-1, 1)
        if self.unsup_calibrator is None:  # pragma: no cover - defensive
            return 1.0 / (1.0 + np.exp(-raw.ravel() * 10.0))
        return self.unsup_calibrator.predict_proba(raw)[:, 1]

    def save(self, path=None):
        import pickle
        from pathlib import Path

        path = Path(path) if path else Path(__file__).resolve().parents[2] / "artifacts" / "fraud" / "fraud_model.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(self, fh)
        return path

    @classmethod
    def load(cls, path=None) -> "FraudModel":
        import pickle
        from pathlib import Path

        path = Path(path) if path else Path(__file__).resolve().parents[2] / "artifacts" / "fraud" / "fraud_model.pkl"
        with path.open("rb") as fh:
            return pickle.load(fh)  # noqa: S301 - trusted local artifact


def train_fraud(*, n: int = 20000, random_state: int = 7) -> tuple[FraudModel, MetricsReport]:
    from imblearn.over_sampling import SMOTE
    from sklearn.ensemble import HistGradientBoostingClassifier

    df = synthesize(n=n, seed=random_state)
    X, y = df[list(FEATURES)], df[TARGET].astype(int).values

    X_tr, X_test, y_tr, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=random_state
    )
    X_fit, X_thr, y_fit, y_thr = train_test_split(
        X_tr, y_tr, test_size=0.30, stratify=y_tr, random_state=random_state
    )

    notes = [
        f"Synthetic stream, {len(df):,} transactions, {y.mean():.2%} fraudulent.",
        "Velocity features (txn_count_1h, distinct_merchants_24h) are windowed — a point-in-time "
        "model without them cannot see card-testing bursts.",
    ]

    # SMOTE on the fit fold only.
    k = max(1, min(5, int(y_fit.sum()) - 1))
    X_res, y_res = SMOTE(random_state=random_state, k_neighbors=k).fit_resample(X_fit, y_fit)
    notes.append(
        f"SMOTE on the fit fold only: {int(y_fit.sum())}/{len(y_fit)} positives before -> "
        f"{int(y_res.sum())}/{len(y_res)} after."
    )

    supervised = HistGradientBoostingClassifier(
        max_iter=250, max_depth=6, learning_rate=0.08, random_state=random_state
    ).fit(X_res, y_res)

    # The isolation forest is fit on the *unresampled* fit fold — oversampling would invent
    # fraudulent geometry the unsupervised model should not learn as normal.
    unsupervised = IsolationForest(
        n_estimators=200, contamination=float(y_fit.mean()), random_state=random_state
    ).fit(X_fit)

    # Calibrate the forest's anomaly score into a probability on the *threshold* fold, which
    # the forest has not seen. Using the fit fold would be optimistic; using the test fold
    # would leak into the reported metrics.
    from sklearn.linear_model import LogisticRegression

    calibrator = LogisticRegression(max_iter=1000).fit(
        unsupervised.decision_function(X_thr).reshape(-1, 1), y_thr
    )
    notes.append(
        "The isolation forest's anomaly score is Platt-scaled into a probability on a held-out "
        "fold. The naive logistic squashing this replaces scored an ordinary transaction at "
        "0.94, because it never learned where the normal population sits."
    )

    bundle = FraudModel(
        supervised=supervised,
        unsupervised=unsupervised,
        unsup_calibrator=calibrator,
        feature_names=list(FEATURES),
        trained_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        n_train=len(X_fit),
        base_rate=float(y.mean()),
        notes=notes,
    )

    def combined(X: pd.DataFrame) -> np.ndarray:
        return np.maximum(supervised.predict_proba(X)[:, 1], bundle.unsupervised_proba(X))

    # Fraud ops care about catching fraud at a tolerable false-positive rate, so pick the
    # threshold on the reserved fold by F1 rather than defaulting to 0.5.
    thr_scores = combined(X_thr)
    grid = np.arange(0.05, 0.96, 0.01)
    f1s = [f1_score(y_thr, (thr_scores >= t).astype(int), zero_division=0) for t in grid]
    bundle.threshold = float(grid[int(np.argmax(f1s))])
    notes.append(
        f"Threshold {bundle.threshold:.2f} chosen by F1 on a reserved fold "
        f"(F1={max(f1s):.4f}), not at the default 0.50."
    )

    test_scores = combined(X_test)
    pred = (test_scores >= bundle.threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()

    scores = {
        "pr_auc": float(average_precision_score(y_test, test_scores)),
        "roc_auc": float(roc_auc_score(y_test, test_scores)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "baseline_pr_auc": float(y_test.mean()),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
    }
    bundle.metrics = scores
    notes.append(
        f"PR-AUC {scores['pr_auc']:.3f} vs {scores['baseline_pr_auc']:.4f} baseline "
        f"({scores['pr_auc']/scores['baseline_pr_auc']:.1f}x lift). At a "
        f"{y_test.mean():.2%} positive rate, accuracy would be "
        f"{1 - y_test.mean():.1%} for a model that flags nothing — which is why it is not reported."
    )

    notes.append(
        "Known limitation: mode-B account takeover — an attacker with the PIN, an aged card, "
        "daytime hours and normal spend — scores near zero. It is not detectable from "
        "transaction features alone; catching it needs device fingerprinting or behavioural "
        "signals this body does not have. The recall figure above already reflects that miss."
    )

    report = MetricsReport(
        n_samples=int(len(X_test)),
        positive_rate=float(y_test.mean()),
        scores=scores,
        confusion={"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        threshold=bundle.threshold,
        notes=notes,
    )
    return bundle, report


if __name__ == "__main__":  # pragma: no cover
    import json

    b, r = train_fraud()
    b.save()
    print(json.dumps(r.model_dump(), indent=2))
