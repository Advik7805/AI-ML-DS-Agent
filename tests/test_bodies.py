"""Tests for the non-churn bodies, and for the agent that reaches them.

Deliberately assert *invariants* rather than exact metric values: these models train on
generated data, so pinning a number would make the suite brittle without making the model
better. What must hold is direction, calibration sanity, and honesty about limits.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.contract import ModuleRegistry


@pytest.fixture(scope="module")
def registry() -> ModuleRegistry:
    reg = ModuleRegistry().discover("modules")
    assert getattr(reg, "load_errors", {}) == {}
    return reg


# --- RAG ---------------------------------------------------------------------


def test_rag_retrieves_the_right_passage(registry) -> None:
    rag = registry.get("rag")
    res = rag.call_tool("search_knowledge", {"query": "What discount can an agent approve?", "k": 3})
    assert res["n_results"] > 0
    assert res["results"][0]["source"] == "retention-playbook"
    assert "20%" in res["results"][0]["snippet"], "the concession table should be in the top passage"


def test_rag_returns_citations(registry) -> None:
    rag = registry.get("rag")
    res = rag.call_tool("search_knowledge", {"query": "refund window", "k": 3})
    assert "refund-policy" in res["citation"]
    assert "Sources" in res["citation"]


def test_rag_ranks_by_relevance(registry) -> None:
    rag = registry.get("rag")
    res = rag.call_tool("search_knowledge", {"query": "uptime availability credit", "k": 5})
    scores = [r["score"] for r in res["results"]]
    assert scores == sorted(scores, reverse=True)
    assert res["results"][0]["source"] == "sla"


def test_rag_is_honest_about_no_match(registry) -> None:
    rag = registry.get("rag")
    pred = rag.predict({"query": "xyzzy quantum flux capacitor"})
    assert pred.label == "No grounding"
    assert "from memory" in pred.decision


def test_rag_does_not_ground_on_a_single_shared_word(registry) -> None:
    """TF-IDF cosine can be driven entirely by one token.

    "What is the airspeed velocity of an unladen swallow" matched the fraud protocol on the
    word "velocity" alone and scored 0.175 — confident-looking and completely unrelated.
    """
    rag = registry.get("rag")
    pred = rag.predict({"query": "what is the airspeed velocity of an unladen swallow"})
    assert pred.label == "Weak grounding", (
        f"got '{pred.label}' at score {pred.score} — a one-word match is not grounding"
    )
    assert pred.meta["matched_terms"] < 2


def test_rag_evaluate_reports_retrieval_quality(registry) -> None:
    m = registry.get("rag").evaluate()
    assert m.scores["corpus_documents"] >= 6
    assert m.scores["corpus_passages"] >= 12
    assert m.scores["hit@1"] >= 0.6
    assert any("smoke test" in n for n in m.notes), "must not oversell a hand-written probe set"


# --- Fraud -------------------------------------------------------------------


def test_fraud_separates_card_testing_from_a_coffee(registry) -> None:
    fraud = registry.get("fraud")
    loud = fraud.call_tool("score_transaction", {
        "amount": 4200, "hour_of_day": 3, "distance_from_home_km": 1500, "txn_count_1h": 12,
        "distinct_merchants_24h": 14, "card_age_days": 45, "avg_amount_30d": 60,
        "is_night": 1, "is_foreign": 1, "pin_used": 0,
    })
    benign = fraud.call_tool("score_transaction", {
        "amount": 5, "hour_of_day": 13, "distance_from_home_km": 2, "txn_count_1h": 1,
        "distinct_merchants_24h": 2, "card_age_days": 900, "avg_amount_30d": 45,
        "is_night": 0, "is_foreign": 0, "pin_used": 1,
    })
    assert loud["score"] > 0.9
    assert benign["score"] < 0.1, (
        f"an ordinary coffee scored {benign['score']} — the unsupervised score is miscalibrated again"
    )


def test_fraud_unsupervised_score_is_calibrated(registry) -> None:
    """Regression test for the 1/(1+exp(score*6)) bug that scored normal points at 0.94."""
    from modules.fraud.data import FEATURES, synthesize

    fraud = registry.get("fraud")
    df = synthesize(n=3000, seed=4)
    legit = df[df["is_fraud"] == 0].sample(150, random_state=0)

    scores = [
        fraud.bundle.unsupervised_proba(fraud.bundle.frame({f: float(r[f]) for f in FEATURES}))[0]
        for _, r in legit.iterrows()
    ]
    mean = float(np.mean(scores))
    assert mean < 0.35, f"mean unsupervised score on legitimate traffic is {mean:.3f} — not calibrated"


def test_fraud_reports_the_account_takeover_blind_spot(registry) -> None:
    """The model genuinely cannot catch quiet account takeover. It must say so."""
    m = registry.get("fraud").call_tool("get_fraud_model_metrics", {})
    assert any("account takeover" in n for n in m["notes"])
    assert m["scores"]["recall"] < 0.9, "a recall this high on overlapping data would be suspicious"


def test_fraud_stream_is_labelled_as_enriched(registry) -> None:
    """A fraud-enriched demo feed that does not say so is quietly misleading."""
    s = registry.get("fraud").call_tool("summarise_fraud_stream", {"n": 40})
    assert s["enriched"] is True
    assert "enrichment_note" in s
    assert s["n_flagged"] > 0, "the demo feed should actually show something"


def test_fraud_metrics_are_not_impossibly_good(registry) -> None:
    """PR-AUC 1.000 was the symptom of a generator with no class overlap."""
    m = registry.get("fraud").call_tool("get_fraud_model_metrics", {})
    assert m["scores"]["pr_auc"] < 0.95, "perfect separation means the data is not realistic"
    assert m["scores"]["pr_auc"] > 10 * m["scores"]["baseline_pr_auc"]


# --- Maintenance -------------------------------------------------------------


def test_maintenance_ranks_engines(registry) -> None:
    maint = registry.get("maintenance")
    res = maint.call_tool("rank_engines_by_urgency", {"n": 5})
    assert res["n_engines"] == 100
    ruls = [e["predicted_rul"] for e in res["engines"]]
    assert ruls == sorted(ruls), "must be ordered by urgency, i.e. ascending RUL"
    assert all(0 <= e["predicted_rul"] <= 125 for e in res["engines"])


def test_maintenance_error_is_reported_by_band(registry) -> None:
    """Overall MAE hides where the model fails; the band breakdown is the point."""
    m = registry.get("maintenance").call_tool("get_maintenance_model_metrics", {})
    band_keys = [k for k in m["scores"] if k.startswith("mae_") and k != "mae_cycles"]
    assert len(band_keys) == 3
    assert m["scores"]["r2"] > 0.5, "R^2 near zero means the degradation signal is unlearnable"


def test_maintenance_rul_respects_the_cap(registry) -> None:
    from modules.maintenance.data import FEATURES, RUL_CAP, synthesize

    maint = registry.get("maintenance")
    df = synthesize(n_units=3)
    row = df.iloc[0]
    payload = {f: float(row[f]) for f in FEATURES if f in df.columns}
    pred = maint.predict(payload)
    assert 0 <= pred.meta["rul_cycles"] <= RUL_CAP
    assert 0.0 <= pred.score <= 1.0


def test_maintenance_uses_grouped_cv(registry) -> None:
    m = registry.get("maintenance").call_tool("get_maintenance_model_metrics", {})
    assert any("GroupKFold" in n for n in m["notes"]), "random splits leak across engine cycles"


# --- Pricing -----------------------------------------------------------------


def test_pricing_recovers_true_elasticity(registry) -> None:
    """Only verifiable because the demand data is generated — and that is worth checking."""
    m = registry.get("pricing").evaluate()
    assert m.scores["mae_elasticity"] < 0.4
    assert abs(m.scores["bias_elasticity"]) < 0.2
    assert m.scores["share_identified"] > 0.7


def test_pricing_never_extrapolates_beyond_observed_prices(registry) -> None:
    """The safety rail that stops a pricing model destroying demand."""
    pricing = registry.get("pricing")
    for sku in list(pricing.model.coefficients)[:6]:
        meta = pricing.predict({"sku": sku}).meta
        lo, hi = meta["observed_price_range"]
        assert lo <= meta["recommended_price"] <= hi, f"{sku}: {meta['recommended_price']} outside [{lo}, {hi}]"


def test_pricing_flags_unidentified_elasticity(registry) -> None:
    res = registry.get("pricing").call_tool("list_skus_by_opportunity", {"n": 5})
    assert res["n_skus"] >= 20
    assert "not_identified" in res
    assert res["note"], "must explain what 'not identified' implies"


def test_pricing_unknown_sku_raises(registry) -> None:
    with pytest.raises(ValueError, match="Unknown SKU"):
        registry.get("pricing").predict({"sku": "SKU-DOES-NOT-EXIST"})


def test_pricing_carries_a_confidence_interval(registry) -> None:
    meta = registry.get("pricing").predict({"sku": "SKU-001"}).meta
    ci = meta["elasticity_ci"]
    assert len(ci) == 2 and ci[0] < ci[1]
    assert ci[0] <= meta["elasticity"] <= ci[1]


# --- Diagnostics -------------------------------------------------------------


def test_diagnostics_uses_the_real_pima_dataset(registry) -> None:
    m = registry.get("xai_health").evaluate()
    assert m.positive_rate > 0


def test_diagnostics_runs_a_fairness_audit(registry) -> None:
    audit = registry.get("xai_health").call_tool("audit_model_fairness", {})
    assert audit["groups"], "audit must return per-group numbers"
    assert audit["worst_served_group"] is not None
    assert audit["fnr_gap"] is not None
    assert "false-negative rate" in audit["metric_rationale"].lower()


def test_diagnostics_states_the_grouping_is_a_proxy(registry) -> None:
    """A bias audit against a fabricated group would look like evidence and mislead."""
    audit = registry.get("xai_health").call_tool("audit_model_fairness", {})
    assert "proxy" in audit["grouping_variable"].lower()


def test_diagnostics_reports_groups_through_the_contract(registry) -> None:
    m = registry.get("xai_health").evaluate()
    assert m.groups, "groups should be part of MetricsReport, not an ad-hoc attribute"
    assert m.group_basis, "a group breakdown without a stated basis is not interpretable"


def test_diagnostics_missing_values_are_handled(registry) -> None:
    """Zero is a missing sentinel in Pima; passing null must not blow up."""
    mod = registry.get("xai_health")
    pred = mod.predict({
        "pregnancies": 3, "plasma_glucose": None, "blood_pressure": 72,
        "triceps_skin_thickness": None, "insulin": None, "bmi": 33,
        "diabetes_pedigree": 0.6, "age": 45,
    })
    assert 0.0 <= pred.score <= 1.0
    assert pred.explanation is not None
    assert any("median-imputed" in c for c in pred.explanation.caveats)


def test_diagnostics_carries_a_not_a_device_warning(registry) -> None:
    pred = registry.get("xai_health").predict({
        "pregnancies": 2, "plasma_glucose": 160, "blood_pressure": 70, "triceps_skin_thickness": 30,
        "insulin": 200, "bmi": 38, "diabetes_pedigree": 0.9, "age": 55,
    })
    assert "not a diagnostic device" in str(pred.meta).lower()


# --- Scraper -----------------------------------------------------------------


def test_scraper_mock_is_deterministic(registry) -> None:
    s = registry.get("scraper")
    a = s.call_tool("scrape_url", {"url": "https://example.com/x"})["markdown"]
    b = s.call_tool("scrape_url", {"url": "https://example.com/x"})["markdown"]
    assert a == b


def test_scraper_rejects_non_http(registry) -> None:
    with pytest.raises(ValueError, match="http"):
        registry.get("scraper").predict({"url": "ftp://example.com/x"})


def test_scraper_live_mode_respects_the_allowlist(registry, monkeypatch) -> None:
    """The rail that stops prompt injection pointing the agent at an arbitrary host."""
    monkeypatch.setenv("NEXUS_SCRAPER_LIVE", "1")
    monkeypatch.setenv("NEXUS_SCRAPER_ALLOW", "allowed.test")
    s = registry.get("scraper")
    res = s.call_tool("scrape_url", {"url": "https://not-allowed.test/page"})
    assert res["fetcher"] == "blocked"
    assert "NEXUS_SCRAPER_ALLOW" in res["error"]


def test_scraper_states_its_production_backend(registry) -> None:
    caps = registry.get("scraper").call_tool("scraper_capabilities", {})
    assert caps["live"] is False
    assert "Firecrawl" in caps["production_backend"]


# --- Agent -------------------------------------------------------------------


def test_agent_reaches_every_body(registry) -> None:
    from core.agent.graph import HubAgent
    from core.agent.providers import MockProvider

    agent = HubAgent(registry, MockProvider())
    queries = {
        "get_churn_model_metrics": "What are the churn model metrics?",
        "rank_at_risk_customers": "Rank the top 5 customers most likely to churn",
        "get_fraud_model_metrics": "How good is the fraud model?",
        "summarise_fraud_stream": "What's happening on the fraud feed right now?",
        "search_knowledge": "Search the knowledge base for the refund window",
        "rank_engines_by_urgency": "Which engines need maintenance first?",
        "optimise_price": "Optimise the price for SKU-003",
        "list_skus_by_opportunity": "Which SKUs have the biggest pricing opportunity?",
        "audit_model_fairness": "Audit the diabetes model for fairness",
        "scrape_url": "Can you scrape a competitor page?",
    }
    for expected_tool, query in queries.items():
        result = agent.run(query)
        assert expected_tool in result["tools_used"], (
            f"'{query}' routed to {result['tools_used']}, expected {expected_tool}"
        )
        assert result["answer"], f"no answer for '{query}'"


def test_agent_answers_without_tools_when_none_fit(registry) -> None:
    from core.agent.graph import HubAgent
    from core.agent.providers import MockProvider

    result = HubAgent(registry, MockProvider()).run("hello there")
    assert result["tools_used"] == []
    assert result["answer"]


def test_agent_trace_records_every_step(registry) -> None:
    from core.agent.graph import HubAgent
    from core.agent.providers import MockProvider

    result = HubAgent(registry, MockProvider()).run("What are the churn model metrics?")
    nodes = [t["node"] for t in result["trace"]]
    assert "plan" in nodes and "act" in nodes
    assert all("ms" in t for t in result["trace"]), "the UI renders timings, so they must exist"


def test_agent_iteration_cap_prevents_a_runaway_loop(registry) -> None:
    """A provider that always calls a tool must be stopped, not allowed to spin."""
    from core.agent.graph import HubAgent
    from core.agent.providers import Message, ToolCall

    class RunawayProvider:
        name = "runaway"

        def complete(self, messages, tools):
            return Message(role="assistant", tool_calls=[
                ToolCall(id="c", name="list_knowledge_sources", arguments={})
            ])

    agent = HubAgent(registry, RunawayProvider(), max_iterations=3)
    result = agent.run("loop forever")
    acts = [t for t in result["trace"] if t.get("node") == "act"]
    assert len(acts) == 3, f"expected the cap to stop it at 3, got {len(acts)}"
    assert result["answer"]


def test_agent_survives_a_tool_that_raises(registry) -> None:
    """One body failing must not take down the conversation."""
    from core.agent.graph import HubAgent
    from core.agent.providers import MockProvider

    result = HubAgent(registry, MockProvider()).run("Optimise the price for SKU-999")
    assert result["answer"], "the agent should still answer after a tool error"
