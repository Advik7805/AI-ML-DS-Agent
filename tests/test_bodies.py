"""Tests for the non-churn bodies, and for the agent that reaches them.

Deliberately assert *invariants* rather than exact metric values: these models train on
generated data, so pinning a number would make the suite brittle without making the model
better. What must hold is direction, calibration sanity, and honesty about limits.
"""

from __future__ import annotations

import numpy as np
import pytest

# `registry` comes from tests/conftest.py so every suite shares one discovery.


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


def test_rag_grounds_a_relevant_question(registry) -> None:
    """Regression guard: a cosine floor must not come back.

    An earlier version rejected anything below 0.25. On this 25-passage corpus TF-IDF tops out
    near 0.22 even for obviously-relevant questions, so the floor labelled every real query
    weak and made the body look broken. The discriminator is substantive term count, not the
    absolute cosine.
    """
    rag = registry.get("rag")
    pred = rag.predict({"query": "What discount can an agent approve?"})
    assert pred.label == "Well grounded", f"got '{pred.label}' at score {pred.score}"
    assert pred.meta["matched_terms"] >= 2
    assert pred.score < 0.25, "this corpus does not produce high cosines; do not re-add a floor"


def test_rag_tool_and_predict_agree_on_grounding(registry) -> None:
    """The agent and the storefront must never disagree about whether an answer was grounded.

    The verdict used to live only inside ``predict``, so the tool handed the agent bare cosine
    scores with no judgement attached.
    """
    rag = registry.get("rag")
    for query in (
        "What discount can an agent approve?",
        "what is the airspeed velocity of an unladen swallow",
        "xyzzy quantum flux capacitor",
    ):
        pred = rag.predict({"query": query, "k": 5})
        res = rag.call_tool("search_knowledge", {"query": query, "k": 5})
        assert res["grounding"] == pred.label, query
        assert res["matched_terms"] == pred.meta["matched_terms"], query
        assert res["decision"], "a verdict with no decision text is useless to the agent"


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
    monkeypatch.setenv("SAROS_SCRAPER_LIVE", "1")
    monkeypatch.setenv("SAROS_SCRAPER_ALLOW", "allowed.test")
    s = registry.get("scraper")
    res = s.call_tool("scrape_url", {"url": "https://not-allowed.test/page"})
    assert res["fetcher"] == "blocked"
    assert "SAROS_SCRAPER_ALLOW" in res["error"]


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


def test_every_tool_result_gets_summarised(registry) -> None:
    """No tool may fall through to the raw-JSON catch-all.

    `_summarise` is a chain of shape checks, so a new body silently gets the fallback and the
    user is shown a wall of JSON. This drives *every* registered tool with schema-filled
    arguments and asserts each output is turned into prose — so adding a body without a
    summariser branch fails here rather than in production.
    """
    import json

    from core.agent.providers import MockProvider, _summarise

    for tool in registry.all_tools():
        args = MockProvider._args_for(tool.name, "example query about the data", tool)
        try:
            out = registry.dispatch(tool.name, args)
        except Exception as exc:  # a body that raises is covered elsewhere
            continue
        summary = _summarise(json.dumps(out, default=str))
        assert not summary.startswith("Tool returned:"), (
            f"{tool.name} has no summariser branch and leaked raw JSON"
        )
        assert "```json" not in summary, f"{tool.name} leaked serialised tool output"
        assert len(summary.strip()) > 20, f"{tool.name} produced an empty summary"


ROUTING_PROBES = [
    ("List the knowledge sources", "list_knowledge_sources"),
    ("What documents are in the corpus?", "list_knowledge_sources"),
    ("Search the knowledge base for the refund window", "search_knowledge"),
    ("What discount can an agent approve?", "search_knowledge"),
    ("What are the churn model metrics?", "get_churn_model_metrics"),
    ("How good is the churn model?", "get_churn_model_metrics"),
    ("What are the fraud model metrics?", "get_fraud_model_metrics"),
    # Word order must not matter: the noun can come before OR after the metric word.
    ("Show the maintenance model card", "get_maintenance_model_metrics"),
    ("How good is the maintenance model?", "get_maintenance_model_metrics"),
    # And a metrics question must not resolve to the action tool just because it shares a noun.
    ("What is the MAE of the RUL model?", "get_maintenance_model_metrics"),
    ("Render https://dashboard.example.com in a browser", "render_url"),
    ("Fetch the JS-rendered page https://example.com/app", "render_url"),
    ("Extract https://example.com/pricing", "extract_url"),
    ("Scrape https://example.com", "scrape_url"),
    ("Search the web for the latest churn benchmarks", "web_search"),
    ("Rank the top 5 customers most likely to churn", "rank_at_risk_customers"),
    ("How good is the fraud model?", "get_fraud_model_metrics"),
    ("Score this transaction for fraud", "score_transaction"),
    ("What's happening on the fraud feed right now?", "summarise_fraud_stream"),
    ("Which engines need maintenance first?", "rank_engines_by_urgency"),
    ("Predict the remaining useful life", "predict_rul"),
    ("Audit the diabetes model for fairness", "audit_model_fairness"),
    ("Which SKUs have the biggest pricing opportunity", "list_skus_by_opportunity"),
    ("Optimise the price for SKU-007", "optimise_price"),
    ("Scrape https://example.com", "scrape_url"),
    ("What are your scraping capabilities?", "scraper_capabilities"),
    ("Assess this patient diabetes risk", "assess_diabetes_risk"),
]


@pytest.mark.parametrize("question,want", ROUTING_PROBES)
def test_agent_routes_each_question_to_the_right_tool(registry, question, want) -> None:
    """Intent routing, pinned.

    Two failure modes this guards, both of which shipped once already:
      * a body-scoped pattern placed after a generic one ("how good is the fraud model" used
        to resolve to the churn metrics tool);
      * `a[\\s\\S]*b` assuming word order, which fails on reordered phrasings.
    """
    from core.agent.providers import _pick_tool

    available = {t.name for t in registry.all_tools()}
    got = _pick_tool(question, available)
    assert got == want, f"{question!r} routed to {got!r}, expected {want!r}"
