"""Gateway tests — exercise the real FastAPI app over HTTP, not the module directly.

These are the tests that prove a *new* body gets an API for free: every endpoint here is
generic and keyed on ``{module_id}``, with zero churn-specific code in the gateway.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from apps.api.main import app, registry


@pytest.fixture(scope="module")
def client():
    if len(registry) == 0:
        registry.discover("modules")
    with TestClient(app) as c:
        yield c


RECORD = {
    "account_length": 120,
    "number_vmail_messages": 0,
    "total_day_minutes": 180,
    "total_day_calls": 110,
    "total_eve_minutes": 200,
    "total_eve_calls": 100,
    "total_night_minutes": 200,
    "total_night_calls": 100,
    "total_intl_minutes": 10,
    "total_intl_calls": 4,
    "number_customer_service_calls": 1,
    "international_plan": 0,
    "voice_mail_plan": 0,
}


def test_health_reports_loaded_modules(client) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "churn" in body["modules_loaded"]
    assert body["load_errors"] == {}


def test_modules_list_is_contract_shaped(client) -> None:
    r = client.get("/api/v1/modules")
    assert r.status_code == 200
    mods = r.json()
    assert len(mods) >= 1
    churn = next(m for m in mods if m["id"] == "churn")
    assert churn["status"] == "live"
    assert churn["routes"]["ui"] == "/churn"
    assert churn["accent"].startswith("#")
    assert len(churn["tools"]) == 3


def test_predict_endpoint(client) -> None:
    r = client.post("/api/v1/modules/churn/predict", json={"features": RECORD})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["module_id"] == "churn"
    assert 0.0 <= body["score"] <= 1.0
    assert body["label"] and body["decision"]
    assert body["factors"], "top drivers should be inlined for card rendering"
    assert body["meta"]["threshold"] > 0


def test_predict_rejects_incomplete_features_with_422(client) -> None:
    r = client.post("/api/v1/modules/churn/predict", json={"features": {"account_length": 10}})
    assert r.status_code == 422
    assert "Missing required features" in r.json()["detail"]


def test_unknown_module_is_404(client) -> None:
    assert client.post("/api/v1/modules/nope/predict", json={"features": {}}).status_code == 404
    assert client.get("/api/v1/modules/nope").status_code == 404


def test_explain_endpoint(client) -> None:
    r = client.post("/api/v1/modules/churn/explain", json={"features": {**RECORD, "number_customer_service_calls": 7}})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["kind"] == "shap"
    assert len(body["attributions"]) == 13
    mags = [abs(a["attribution"]) for a in body["attributions"]]
    assert mags == sorted(mags, reverse=True)
    assert all(a["direction"] in {"increases_risk", "decreases_risk", "neutral"} for a in body["attributions"])


def test_metrics_endpoint(client) -> None:
    r = client.get("/api/v1/modules/churn/metrics")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n_samples"] > 0
    assert body["scores"]["pr_auc"] > body["scores"]["baseline_pr_auc"]
    assert sum(body["confusion"].values()) == body["n_samples"]
    assert body["notes"], "methodological notes must be exposed, not hidden in a notebook"


def test_agent_tool_registry_is_llm_ready(client) -> None:
    """Tools must carry JSON Schema so the hub can bind them onto an LLM with no glue."""
    r = client.get("/api/v1/agent/tools")
    assert r.status_code == 200
    tools = r.json()["tools"]
    names = {t["name"] for t in tools}
    assert {"score_customer_churn", "rank_at_risk_customers", "get_churn_model_metrics"} <= names
    for t in tools:
        assert t["description"]
        assert t["parameters_schema"]["type"] == "object"


def test_agent_dispatch_routes_to_the_owning_body(client) -> None:
    r = client.post("/api/v1/agent/tools/rank_at_risk_customers", json={"arguments": {"n": 3, "min_score": 0.0}})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["module_id"] == "churn"
    assert body["result"]["n_returned"] == 3
    scores = [c["score"] for c in body["result"]["customers"]]
    assert scores == sorted(scores, reverse=True)


def test_agent_dispatch_unknown_tool_is_404(client) -> None:
    r = client.post("/api/v1/agent/tools/no_such_tool", json={"arguments": {}})
    assert r.status_code == 404


def test_cors_allows_the_frontend_origin(client) -> None:
    r = client.options(
        "/api/v1/modules/churn/predict",
        headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "POST"},
    )
    assert r.status_code in (200, 204)
    assert r.headers.get("access-control-allow-origin") in ("*", "http://localhost:3000")


def test_the_agent_endpoint_reports_degraded_instead_of_claiming_live(client, monkeypatch) -> None:
    """Three states, not two: live, degraded, and no key at all.

    ``live: true`` is read as "a language model is reasoning about this" — so reporting it while
    a template answers is the same lie as the old "Groq is live" badge next to a mock chat. A
    deployment whose provider has failed must say so, with the vendor's own sentence, or the
    person reading it cannot tell a bad answer from a broken deployment.
    """
    import apps.api.main as api
    from core.agent.graph import HubAgent
    from core.agent.providers import FallbackProvider, Message, MockProvider

    original = api.agent
    assert original is not None

    def info_for(provider) -> dict:
        monkeypatch.setattr(api, "agent", HubAgent(original.registry, provider))
        return client.get("/api/v1/agent").json()

    class WorkingGroq:
        """A live provider that answers — the state the badge is *supposed* to call live."""

        name = "groq"
        model = "openai/gpt-oss-120b"

        def complete(self, messages, tools):  # noqa: ANN001, ANN201 - protocol shape
            return Message(role="assistant", content="answered")

    class RetiredGroq(WorkingGroq):
        """The deploy from the screenshot: key present, model retired, 404 on every call."""

        model = "llama-3.3-70b-versatile"

        def complete(self, messages, tools):  # noqa: ANN001, ANN201
            raise RuntimeError(
                'Groq 404: {"error":{"message":"The model llama-3.3-70b-versatile does not exist '
                'or you do not have access to it.","type":"invalid_request_error",'
                '"code":"model_not_found"}}'
            )

    live = info_for(FallbackProvider(WorkingGroq()))
    assert live["live"] is True and live["degraded"] is False and live["note"] is None
    assert live["provider_model"] == "openai/gpt-oss-120b"

    mock = info_for(MockProvider())
    assert mock["live"] is False and mock["degraded"] is False
    assert "no LLM key is configured" in mock["note"]

    degraded_provider = FallbackProvider(RetiredGroq(), backup=MockProvider())
    # One answer through it is enough to fail over; the endpoint then has to report that state.
    degraded_provider.complete([Message(role="user", content="rank the at-risk customers")], [])
    degraded = info_for(degraded_provider)

    assert degraded["live"] is False, "a template is answering; the badge must not say live"
    assert degraded["degraded"] is True
    assert degraded["provider_model"] == "llama-3.3-70b-versatile"
    assert "groq failed" in degraded["note"].lower()
    assert "does not exist" in degraded["note"]
    # The note is for a person: no JSON envelope, no exception class.
    assert "{" not in degraded["note"] and "RuntimeError" not in degraded["note"]
