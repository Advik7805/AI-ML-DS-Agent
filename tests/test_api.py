"""Gateway tests — exercise the real FastAPI app over HTTP, not the module directly.

These are the tests that prove a *new* body gets an API for free: every endpoint here is
generic and keyed on ``{module_id}``, with zero churn-specific code in the gateway.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from threading import Event

import pytest
from fastapi.testclient import TestClient

import apps.api.main as api_main
from apps.api.main import app, registry
from core.agent.providers import MockProvider


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
    assert body["training"]["status"] in {"not_started", "starting", "running", "ready", "partial", "failed"}


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


def test_agent_chat_stream_reports_real_progress_updates(client) -> None:
    assert api_main.agent is not None
    api_main.agent.provider = MockProvider()

    response = client.post("/api/v1/agent/chat/stream", json={"message": "Say hello."})

    assert response.status_code == 200, response.text
    assert "event: plan" in response.text
    assert "event: done" in response.text
    assert '"answer"' in response.text


def test_slow_stream_does_not_block_health(client, monkeypatch) -> None:
    assert api_main.agent is not None
    entered = Event()
    release = Event()
    health_completed = Event()

    def slow_stream(_message, history=None):
        entered.set()
        release.wait(timeout=3)
        yield {"node": "done", "answer": "ready"}

    monkeypatch.setattr(api_main.agent, "stream", slow_stream)

    def get_health():
        response = client.get("/health")
        health_completed.set()
        return response

    with ThreadPoolExecutor(max_workers=2) as pool:
        stream_request = pool.submit(client.post, "/api/v1/agent/chat/stream", json={"message": "wait"})
        try:
            assert entered.wait(timeout=1), "stream generator did not start"
            health_request = pool.submit(get_health)
            assert health_completed.wait(timeout=0.5), "/health was blocked by synchronous agent work"
            assert health_request.result().status_code == 200
        finally:
            release.set()
        response = stream_request.result(timeout=2)

    assert response.status_code == 200
    assert "event: done" in response.text


def test_health_remains_responsive_during_background_warmup(client, monkeypatch) -> None:
    """A slow model warm-up runs away from both the event loop and /health."""
    original_thread = api_main._training_thread
    if original_thread is not None and original_thread.is_alive():
        original_thread.join(timeout=10)
    with api_main._training_lock:
        original_state = deepcopy(api_main._training_state)

    started = Event()
    release = Event()

    def blocked_warmup() -> None:
        with api_main._training_lock:
            api_main._training_state.update(
                status="running",
                trainable=["churn"],
                bodies={"churn": {"status": "training"}},
            )
        started.set()
        release.wait(timeout=5)
        with api_main._training_lock:
            api_main._training_state["bodies"]["churn"] = {"status": "ready"}
            api_main._training_state["status"] = "ready"

    monkeypatch.setattr(api_main, "_warm_train_bodies", blocked_warmup)
    with api_main._training_lock:
        api_main._training_state.clear()
        api_main._training_state.update(status="not_started", bodies={})
        api_main._training_thread = None

    async def check_health_while_warming():
        async with api_main.lifespan(app):
            assert await asyncio.to_thread(started.wait, 1), "background warm-up did not start"
            return await asyncio.to_thread(client.get, "/health")

    try:
        response = asyncio.run(check_health_while_warming())
    finally:
        release.set()
        current_thread = api_main._training_thread
        if current_thread is not None:
            current_thread.join(timeout=2)
        with api_main._training_lock:
            api_main._training_state.clear()
            api_main._training_state.update(original_state)
            api_main._training_thread = original_thread

    assert response.status_code == 200
    assert response.json()["training"]["status"] == "running"


def test_cors_allows_the_frontend_origin(client) -> None:
    r = client.options(
        "/api/v1/modules/churn/predict",
        headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "POST"},
    )
    assert r.status_code in (200, 204)
    assert r.headers.get("access-control-allow-origin") in ("*", "http://localhost:3000")
