"""Warm-up tests — the first user request must not be the one that trains the models.

Bodies train lazily on first use. On a free tier that means cold start *plus* training on the
first tool call (a real trace showed 14.5 s). The lifespan hook therefore warms every body on a
background thread, `/health` must never wait for it, and the outcome has to be published so a
slow first request can be diagnosed instead of guessed at.
"""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from apps.api.main import app, registry
from core import warmup


@pytest.fixture(scope="module")
def client():
    if len(registry) == 0:
        registry.discover("modules")
    with TestClient(app) as c:
        yield c


def test_health_never_waits_for_training(client) -> None:
    """A health check that blocks on model training fails the deploy. It must stay instant."""
    started = time.perf_counter()
    r = client.get("/health")
    elapsed = time.perf_counter() - started
    assert r.status_code == 200
    assert elapsed < 1.0, f"/health took {elapsed:.2f}s — it is waiting on warm-up"
    body = r.json()
    assert "warmup" in body
    assert body["warmup"]["status"] in ("idle", "warming", "done")


def test_every_body_is_warmable(registry) -> None:
    """Warming uses the same locked property a request would, so it cannot double-train."""
    warmup.reset()
    state = warmup.warm(registry)
    assert state["status"] == "done"
    assert state["seconds"] is not None
    failures = {b: s.get("error") for b, s in state["bodies"].items() if not s.get("ok")}
    assert not failures, f"bodies failed to warm: {failures}"
    assert set(state["bodies"]) == set(registry.ids()), "every body should be accounted for"


def test_warm_state_reports_what_it_loaded(registry) -> None:
    warmup.reset()
    warmup.warm(registry)
    state = warmup.state()
    churn = state["bodies"]["churn"]
    assert churn["ok"] is True
    assert churn["seconds"] >= 0
    # A headline number, so "is it actually ready" is answerable without another request.
    assert churn["headline"].startswith(("pr_auc=", "r2=", "roc_auc=", "rows=")), churn["headline"]
    assert state["bodies"]["rag"]["role"] == "lazy assets", "rag has no training but does have an index"


def test_a_failing_body_is_reported_not_raised() -> None:
    """One broken body must not take the warm-up (and therefore the app) down."""

    class Exploding:
        @property
        def bundle(self):  # noqa: ANN201 - deliberately raises
            raise RuntimeError("artifact is corrupt")

    result = warmup.warm_one(Exploding())
    assert result["ok"] is False
    assert "artifact is corrupt" in result["error"]


def test_warmup_endpoint_publishes_the_state(client) -> None:
    r = client.get("/api/v1/warmup")
    assert r.status_code == 200
    body = r.json()
    assert "state" in body and "bodies" in body["state"]
    assert "churn" in body["trainable"], "the trainable set comes from flows.train_all, not a copy"


def test_warmup_is_idempotent_because_it_uses_the_module_lock(registry) -> None:
    """Running it twice must not go back to training — the artifact is already in memory."""
    first = warmup.warm(registry)
    second = warmup.warm(registry)
    assert first["bodies"]["churn"]["ok"] and second["bodies"]["churn"]["ok"]
    # The second pass reuses the cached bundle, so it cannot be dramatically slower.
    assert second["seconds"] < max(2.0, first["seconds"] * 3 + 2.0)
