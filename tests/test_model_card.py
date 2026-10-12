"""The model card a body shows must be the evaluation that model was actually judged by.

The bug these tests lock out: a body loaded from its cached artifact had no metrics object, so the
service rebuilt one with ``n_samples=0, positive_rate=0.0`` and the UI displayed those zeros as
fact. A wrong number on a model card is worse than a slow one — it is the number a reviewer reads
first. Bundles now carry their report, so a reload restores the evaluation instead of inventing it.
"""

from __future__ import annotations

import pytest

from core.artifacts import SCHEMA, load, save

TRAINABLE = ("churn", "fraud", "maintenance", "xai_health")


@pytest.mark.parametrize("body", TRAINABLE)
def test_the_report_travels_with_the_artifact(body: str, registry, tmp_path) -> None:
    module = registry.get(body)
    report = module.evaluate()  # trains on a cold cache, loads the stored report otherwise
    bundle = module.bundle

    assert getattr(bundle, "report", None) is not None, f"{body}'s bundle must carry its report"
    assert bundle.report.n_samples == report.n_samples > 0, (
        f"{body} reports {bundle.report.n_samples} samples"
    )
    assert bundle.report.scores == report.scores

    # What a second process sees after a restart: same card, no retraining.
    path = tmp_path / f"{body}.pkl"
    save(bundle, path)
    restored = load(path)
    assert restored.report.n_samples == report.n_samples
    assert restored.report.scores == report.scores
    assert restored.report.positive_rate == report.positive_rate


def test_a_reloaded_model_card_is_not_all_zeros(registry) -> None:
    """The specific regression: n_samples/positive_rate must survive a load."""
    module = registry.get("maintenance")
    _ = module.bundle  # force the artifact path
    report = module.evaluate()
    assert report.n_samples > 0
    assert report.positive_rate > 0
    assert report.scores, "a model card with no scores is not a model card"
    assert report.confusion or report.groups or report.notes


def test_artifacts_carry_a_schema_fresh_enough_to_include_the_report() -> None:
    """Schema 3 added `report`. A stale artifact must be rejected, which means retraining."""
    assert SCHEMA >= 3
