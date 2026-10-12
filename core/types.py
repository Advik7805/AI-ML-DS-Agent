"""Shared types for the Saros platform.

These are the *contract* types. Every module ("body") produces a ``Prediction`` and, where
it can, an ``Explanation`` in exactly this shape. That is what lets one React component
render "why did the model say this?" for churn, fraud, maintenance and RAG alike — the
visible proof that the bodies are actually connected rather than merely linked in a nav bar.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

#: Which way a feature pushes the prediction.
Direction = Literal["increases_risk", "decreases_risk", "neutral"]

#: How the attributions were produced. Drives which UI affordances are honest to show.
ExplanationKind = Literal["shap", "attention", "retrieval", "heuristic", "none"]

#: The space attributions live in. This matters and is easy to get wrong: SHAP's
#: ``TreeExplainer`` is *exact* in the model's raw margin (log-odds for classification) but
#: only approximate in probability space. Declaring it lets the UI label the axis correctly
#: instead of implying the bars sum to the displayed probability.
AttributionSpace = Literal["log_odds", "probability", "logits", "unspecified"]


def _direction(attribution: float, *, tol: float = 1e-4) -> Direction:
    if attribution > tol:
        return "increases_risk"
    if attribution < -tol:
        return "decreases_risk"
    return "neutral"


class FeatureAttribution(BaseModel):
    """One feature's signed contribution to a single prediction."""

    feature: str
    value: float = Field(description="The feature's actual value for this record.")
    attribution: float = Field(
        description=(
            "Signed contribution to the prediction. Units are given by the enclosing "
            "Explanation's `space` — usually raw log-odds, not probability."
        )
    )
    direction: Direction

    @classmethod
    def of(cls, feature: str, value: float, attribution: float, *, tol: float = 1e-4) -> "FeatureAttribution":
        return cls(
            feature=feature,
            value=round(float(value), 6),
            attribution=round(float(attribution), 6),
            direction=_direction(attribution, tol=tol),
        )


class Explanation(BaseModel):
    """Model-agnostic explanation. SHAP for tabular, attention for nets, provenance for RAG."""

    kind: ExplanationKind = "none"
    space: AttributionSpace = Field(
        default="unspecified",
        description="Units of `attributions`. Do not imply these sum to the displayed probability unless this is 'probability'.",
    )
    attributions: list[FeatureAttribution] = Field(default_factory=list)
    base_value: float | None = Field(
        default=None,
        description="Model's expected output before any features — the SHAP base value.",
    )
    summary: str = Field(default="", description="One-sentence human/agent-readable takeaway.")
    caveats: list[str] = Field(default_factory=list)

    @property
    def top_driver(self) -> FeatureAttribution | None:
        if not self.attributions:
            return None
        return max(self.attributions, key=lambda a: abs(a.attribution))


class Prediction(BaseModel):
    """The single output shape every module returns."""

    module_id: str
    score: float = Field(ge=0.0, le=1.0, description="Primary risk/probability, 0..1.")
    label: str = Field(description="Human label for the score, e.g. 'High churn risk'.")
    decision: str = Field(description="Recommended action. This is what makes it useful.")
    inputs: dict[str, float] = Field(default_factory=dict, description="Echo of features used.")
    factors: list[FeatureAttribution] = Field(
        default_factory=list, description="Top drivers, inlined so a card can render without a 2nd call."
    )
    explanation: Explanation | None = None
    meta: dict[str, Any] = Field(default_factory=dict)


class AgentTool(BaseModel):
    """A capability a module exposes to the hub agent.

    ``parameters_schema`` is JSON Schema, so the LangGraph hub can bind these straight onto
    an LLM as function-calling tools without any module-specific glue.
    """

    name: str
    description: str
    parameters_schema: dict[str, Any] = Field(default_factory=dict)
    module_id: str = ""

    model_config = {"arbitrary_types_allowed": True}


class DatasetCard(BaseModel):
    """Where a body's numbers come from, and whose they are.

    Every body carries one. It exists because "which company, which file, which rows, which
    key?" is the first question a reviewer asks and the first thing an unexplained dashboard
    fails to answer. A model that cannot say what it was trained on should not be trusted with
    a decision, so the answer is a required field rather than a docstring.
    """

    module_id: str = ""
    company: str = ""
    company_note: str = ""
    business_unit: str = ""
    dataset: str = ""
    grain: str = Field(default="", description="What one row is.")
    key: str = Field(default="", description="Primary/join key, so cross-body joins are possible.")
    target: str = Field(default="", description="What the model predicts, in business words.")
    period: str = ""
    origin: Literal["public-dataset", "generated", "corpus", "uploaded", "none"] = "none"
    origin_detail: str = Field(default="", description="Exactly what the numbers are. Never vague.")
    origin_url: str = ""
    file: str = ""
    rows: int | None = Field(default=None, description="Rows as shipped.")
    observed_rows: int | None = Field(default=None, description="Rows the body actually read, once it has.")
    observed_source: str = Field(default="", description="Real/synthetic marker reported by the loader.")
    observed_basis: str = Field(
        default="",
        description=(
            "What observed_rows counts. A loaded artifact can only report the split it was "
            "trained on, and calling that 'rows read' would overstate what the card knows."
        ),
    )
    system_of_record: str = Field(default="", description="Where this would live in production.")
    why: str = ""
    connect: str = Field(default="", description="How this body links to the others.")
    caveats: list[str] = Field(default_factory=list)


class ModuleManifest(BaseModel):
    """What the hub needs to render a body's tile and route to it."""

    id: str
    title: str
    summary: str
    accent: str = "#6366f1"
    version: str = "0.1.0"
    status: Literal["live", "beta", "planned"] = "planned"
    capabilities: list[str] = Field(default_factory=list)
    routes: dict[str, str] = Field(default_factory=dict, description="e.g. {'ui': '/churn', 'api': '/api/v1/churn'}")
    tools: list[AgentTool] = Field(default_factory=list)
    dataset: "DatasetCard | None" = Field(
        default=None, description="Provenance for this body's data. Rendered on the body page."
    )


class MetricsReport(BaseModel):
    """Evaluation output from ``Module.evaluate``. Same shape for every body."""

    n_samples: int
    positive_rate: float
    scores: dict[str, float] = Field(default_factory=dict)
    confusion: dict[str, int] = Field(default_factory=dict)
    threshold: float = 0.5
    notes: list[str] = Field(default_factory=list)
    #: Per-group breakdown, for any body that audits subpopulations (fairness, segments,
    #: operating regimes). Empty when the body has no meaningful grouping.
    groups: dict[str, dict[str, float]] = Field(default_factory=dict)
    #: How the grouping was defined. Required whenever ``groups`` is non-empty, so a
    #: breakdown can never be reported without saying what it was sliced by.
    group_basis: str = ""
