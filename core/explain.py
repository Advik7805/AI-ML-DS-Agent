"""Explainer selection, in one place.

Every body that ships SHAP used to call `shap.TreeExplainer(model)` directly. That works when
the winning candidate happens to be a tree ensemble — and it is a hard crash when it is not:

    InvalidModelError: Model type not yet supported by TreeExplainer:
    <class 'sklearn.linear_model._logistic.LogisticRegression'>

That is not a hypothetical. Churn selects its estimator by PR-AUC over two candidates, gradient
boosting and logistic regression. On the real telco data the boosting wins by a wide margin
(0.677 vs 0.412), so the tree explainer is always the one that runs and the bug never surfaces.
On the synthetic fallback — the exact path `deploy/Dockerfile.api` takes when
`data/fetch.sh` fails at build time, and what any fresh checkout has before the datasets are
pulled — the linear model can win, and `explain()` then raises instead of returning an
explanation.

So: pick the explainer from the estimator, not from an assumption about which model won.

All three branches are exact or near-exact additive decompositions, so the `space` contract in
`core/types.py` still holds; only the caveat wording differs, and callers get that back in the
returned metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ExplainerChoice:
    """The explainer plus enough provenance to describe it honestly."""

    explainer: Any
    #: One of "tree", "linear", "permutation".
    method: str
    #: Whether the decomposition is exact for this model class.
    exact: bool
    note: str


def _is_tree_model(m: Any) -> bool:
    """True for anything shap's TreeExplainer actually supports."""
    name = type(m).__name__
    # sklearn's tree-ensemble estimators. Checking names rather than importing keeps this free
    # of a hard dependency on any particular sklearn version's internal layout.
    tree_names = {
        "DecisionTreeClassifier", "DecisionTreeRegressor",
        "RandomForestClassifier", "RandomForestRegressor",
        "ExtraTreesClassifier", "ExtraTreesRegressor",
        "GradientBoostingClassifier", "GradientBoostingRegressor",
        "HistGradientBoostingClassifier", "HistGradientBoostingRegressor",
    }
    if name in tree_names:
        return True
    # XGBoost / LightGBM / CatBoost expose the tree interface TreeExplainer expects.
    return hasattr(m, "get_booster") or hasattr(m, "booster_") or hasattr(m, "get_tree_params")


def _is_linear_model(m: Any) -> bool:
    return type(m).__name__ in {
        "LogisticRegression", "LogisticRegressionCV",
        "LinearRegression", "Ridge", "RidgeClassifier", "Lasso", "ElasticNet",
        "SGDClassifier", "SGDRegressor", "LinearSVC", "LinearSVR",
    }


def build_explainer(model: Any, background: Any = None) -> ExplainerChoice:
    """Return an explainer suited to `model`, never assuming it is a tree ensemble.

    `background` is only needed for the linear and permutation branches. For a linear model,
    passing the training frame gives a masker centred on the data rather than on zeros — worth
    doing when the caller has it, and harmless to omit.
    """
    import shap

    if _is_tree_model(model):
        return ExplainerChoice(
            explainer=shap.TreeExplainer(model),
            method="tree",
            exact=True,
            note=(
                "SHAP TreeExplainer: exact additive decomposition of the model's raw margin "
                "(log-odds for a classifier)."
            ),
        )

    if _is_linear_model(model):
        masker = background if background is not None else shap.maskers.Independent(np.zeros((1, model.n_features_in_)))
        return ExplainerChoice(
            explainer=shap.LinearExplainer(model, masker),
            method="linear",
            exact=True,
            note=(
                "SHAP LinearExplainer: exact for this model class. The selected estimator was "
                "linear, not a tree ensemble — attributions are the coefficient contributions "
                "relative to the background, not a tree-path decomposition."
            ),
        )

    # Anything else: model-agnostic sampling. Slower and approximate, but it returns an
    # explanation instead of raising. Explicitly labelled as approximate.
    masker = background if background is not None else shap.utils.sample(
        _as_frame(background) if background is not None else np.zeros((1, model.n_features_in_)), 50
    )
    return ExplainerChoice(
        explainer=shap.PermutationExplainer(lambda X: model.predict_proba(X), masker),
        method="permutation",
        exact=False,
        note=(
            "SHAP PermutationExplainer: model-agnostic and approximate (sampling based). Used "
            "because this estimator has no exact SHAP implementation."
        ),
    )


def _as_frame(x: Any) -> Any:
    return x


def explain_rows(choice: ExplainerChoice, X: Any) -> tuple[np.ndarray, np.ndarray]:
    """Compute attributions and base values, normalising shap's shape inconsistencies.

    TreeExplainer returns (n, features) or (n, features, classes) depending on the estimator;
    LinearExplainer and PermutationExplainer differ again. Every caller in this repo wants the
    same thing: per-feature contributions for the positive class, plus a base value per row.
    """
    vals = choice.explainer.shap_values(X)

    # Some explainers return a list with one array per class.
    if isinstance(vals, (list, tuple)):
        vals = vals[-1]
    vals = np.asarray(vals)
    if vals.ndim == 3:
        vals = vals[:, :, -1]

    base = getattr(choice.explainer, "expected_value", None)
    if base is None:
        base = np.zeros(len(vals))
    base = np.asarray(base)
    if base.ndim > 0:
        base = np.broadcast_to(base.ravel()[-1], (len(vals),))
    else:
        base = np.full(len(vals), float(base))

    return vals, base
