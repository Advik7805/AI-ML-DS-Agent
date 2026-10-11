"""Explainer selection must follow the estimator, not an assumption.

This exists because of a real defect: every body called `shap.TreeExplainer(model)` directly.
That works while a tree ensemble wins model selection — and raises `InvalidModelError` the
moment a linear model does. On the real telco data boosting always wins, so it never surfaced;
on the synthetic fallback (what `deploy/Dockerfile.api` produces when `data/fetch.sh` fails,
and what any fresh checkout has before the datasets are pulled) the linear candidate can win.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from core.explain import build_explainer, explain_rows


@pytest.fixture
def xy() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    X = rng.normal(size=(120, 4))
    y = (X[:, 0] + 0.5 * X[:, 1] + rng.normal(scale=0.3, size=120) > 0).astype(int)
    return X, y


@pytest.mark.parametrize(
    "make, expected",
    [
        (lambda: GradientBoostingClassifier(random_state=0), "tree"),
        (lambda: HistGradientBoostingClassifier(random_state=0), "tree"),
        (lambda: LogisticRegression(max_iter=500), "linear"),
    ],
)
def test_explainer_tracks_the_estimator_class(make, expected, xy) -> None:
    X, y = xy
    model = make().fit(X, y)
    choice = build_explainer(model, background=X)
    assert choice.method == expected
    assert choice.exact is True, "tree and linear decompositions are both exact"


def test_a_linear_model_produces_attributions_rather_than_raising(xy) -> None:
    """The regression itself. TreeExplainer raises InvalidModelError here."""
    X, y = xy
    model = LogisticRegression(max_iter=500).fit(X, y)
    choice = build_explainer(model, background=X)

    vals, base = explain_rows(choice, X[:5])
    assert vals.shape == (5, X.shape[1])
    assert base.shape == (5,)
    assert np.isfinite(vals).all(), "attributions must be finite"


def test_attributions_point_the_right_way(xy) -> None:
    """X[:, 0] is the driver, so its mean attribution should dominate."""
    X, y = xy
    model = LogisticRegression(max_iter=500).fit(X, y)
    vals, _ = explain_rows(build_explainer(model, background=X), X)
    mean_abs = np.abs(vals).mean(axis=0)
    assert mean_abs.argmax() == 0, "the strongest feature should get the largest attributions"


def test_the_churn_synthetic_fallback_does_not_crash(tmp_path) -> None:
    """End to end through the path the Dockerfile takes when the dataset fetch fails.

    Before the fix this raised `InvalidModelError: Model type not yet supported by
    TreeExplainer: LogisticRegression`, because the linear candidate wins PR-AUC on synthetic
    data and the explainer was hard-wired to trees.
    """
    from modules.churn.train import train_churn

    model, report = train_churn(csv_path=tmp_path / "does_not_exist.csv")
    assert report.scores["pr_auc"] > 0

    # The crash was in explain(); asserting it returns attributions is the whole point.
    n_features = len(model.feature_names)
    vals, base = explain_rows(build_explainer(model.raw_model.named_steps["clf"]),
                             np.zeros((1, n_features)))
    assert vals.shape == (1, n_features)
    assert np.isfinite(vals).all()
    assert base.shape == (1,)


def test_both_churn_paths_explain_the_same_shape() -> None:
    """Real and synthetic data must both yield usable explanations."""
    from modules.churn.train import train_churn

    for label, csv in [("real", None), ("synthetic", __import__("pathlib").Path("/nope.csv"))]:
        model, _ = train_churn(csv_path=csv)
        clf = model.raw_model.named_steps["clf"]
        choice = build_explainer(clf)
        vals, _ = explain_rows(choice, np.zeros((1, len(model.feature_names))))
        assert vals.shape[1] == len(model.feature_names), label
        assert choice.method in ("tree", "linear", "permutation"), label
