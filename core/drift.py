"""Feature drift detection.

Every body here is trained once and served indefinitely on an ephemeral container. Nothing
currently notices when the data it is being asked to score stops looking like the data it was
trained on — which is the failure mode where a model stays confident, keeps answering, and is
quietly wrong. For the two bodies marketed as real-time (fraud, maintenance) that is the whole
risk.

Method: Population Stability Index per feature. PSI is the standard here because it is
symmetric, bounded, interpretable, and has well-understood thresholds, unlike a raw mean shift
which says nothing about whether the change matters relative to the feature's own spread.

    PSI = sum over bins of (actual% - expected%) * ln(actual% / expected%)

Bins are cut on the *training* distribution, so the reference frame never moves. That matters:
if you re-bin on the incoming data you are measuring the incoming data against itself and will
never see drift.

The conventional reading, which this module reports rather than hides:

    PSI < 0.1   no significant shift
    0.1 - 0.25  moderate shift, worth watching
    > 0.25      significant shift, the model is being asked about data it has not seen

One honest limitation: PSI on binned marginals sees each feature alone. A shift in the joint
distribution that leaves every marginal intact is invisible to it. That is a real blind spot and
it is named in the output rather than left for someone to discover during an incident.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

#: Conventional PSI thresholds. Reported in the payload, not buried in the code.
MODERATE = 0.10
SIGNIFICANT = 0.25

EPS = 1e-6


def psi_bins(reference: Sequence[float] | np.ndarray, n_bins: int = 10) -> np.ndarray:
    """Cut bin edges on the reference distribution, so the frame of reference never moves."""
    ref = np.asarray(reference, dtype=float)
    ref = ref[np.isfinite(ref)]
    if ref.size == 0:
        return np.array([])

    # Quantile edges, not equal-width: real features are skewed, and equal-width bins put
    # everything in one bucket and leave the rest empty, which makes PSI meaningless.
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, n_bins + 1)))
    if edges.size < 3:
        # A near-constant feature has nothing to drift. Return a degenerate frame rather than
        # fabricating bins.
        return np.array([ref.min() - 0.5, ref.max() + 0.5])
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def population_stability_index(
    reference: Sequence[float] | np.ndarray,
    actual: Sequence[float] | np.ndarray,
    edges: np.ndarray | None = None,
) -> float:
    """PSI of `actual` against `reference`. 0 means identical distributions."""
    ref = np.asarray(reference, dtype=float)
    act = np.asarray(actual, dtype=float)
    ref = ref[np.isfinite(ref)]
    act = act[np.isfinite(act)]
    if ref.size == 0 or act.size == 0:
        return 0.0

    e = psi_bins(ref) if edges is None else edges
    if e.size < 2:
        return 0.0

    ref_counts = np.histogram(ref, bins=e)[0].astype(float)
    act_counts = np.histogram(act, bins=e)[0].astype(float)

    ref_pct = ref_counts / max(ref_counts.sum(), 1.0)
    act_pct = act_counts / max(act_counts.sum(), 1.0)

    # An empty bin would make ln(0/x) explode. This is the standard fix and it is why PSI is
    # never exactly 0 in practice.
    ref_pct = np.clip(ref_pct, EPS, None)
    act_pct = np.clip(act_pct, EPS, None)

    return float(np.sum((act_pct - ref_pct) * np.log(act_pct / ref_pct)))


def verdict(value: float) -> str:
    if value < MODERATE:
        return "no significant shift"
    if value < SIGNIFICANT:
        return "moderate shift — monitor"
    return "significant shift — model is outside its training distribution"


def drift_report(
    reference: dict[str, np.ndarray | Sequence[float]],
    actual: dict[str, np.ndarray | Sequence[float]],
    *,
    top_n: int = 5,
) -> dict[str, Any]:
    """Compare a batch of incoming records against the training distribution.

    `reference` and `actual` map feature name -> values. Only features present in both are
    scored; a feature that appeared or vanished is reported separately, because that is a
    schema problem, not a drift problem, and conflating them would hide it.
    """
    common = sorted(set(reference) & set(actual))
    missing = sorted(set(reference) - set(actual))
    unexpected = sorted(set(actual) - set(reference))

    per_feature: list[dict[str, Any]] = []
    for name in common:
        value = population_stability_index(reference[name], actual[name])
        per_feature.append({
            "feature": name,
            "psi": round(value, 4),
            "verdict": verdict(value),
        })

    per_feature.sort(key=lambda d: d["psi"], reverse=True)
    drifted = [f for f in per_feature if f["psi"] >= SIGNIFICANT]
    watching = [f for f in per_feature if MODERATE <= f["psi"] < SIGNIFICANT]

    # Mean PSI across features, not the max. The max tells you the worst feature; the mean tells
    # you whether the dataset as a whole moved. Both matter and reporting only one is misleading.
    mean_psi = float(np.mean([f["psi"] for f in per_feature])) if per_feature else 0.0

    return {
        "n_features_scored": len(per_feature),
        "mean_psi": round(mean_psi, 4),
        "overall": verdict(mean_psi),
        "n_drifted": len(drifted),
        "n_watching": len(watching),
        "top": per_feature[:top_n],
        "features_missing_from_batch": missing,
        "features_not_in_training": unexpected,
        "thresholds": {"moderate": MODERATE, "significant": SIGNIFICANT},
        "caveats": [
            "PSI is computed on binned marginals, so a shift in the joint distribution that "
            "leaves every marginal unchanged is invisible to it.",
            "Bin edges are cut on the training distribution and never move, which is what makes "
            "the comparison meaningful.",
            "Drift is not automatically bad — the population may legitimately have changed. It "
            "means the model is being asked about data it was not trained on, and its error "
            "estimates no longer apply.",
        ],
    }
