"""Every HTTP route must answer, and no route may 500.

This exists because one did. `GET /api/v1/agent` — the endpoint the chat header reads for its
provider badge — called `os.getenv` in a module that never imported `os`, so it raised NameError
and returned `500 Internal Server Error`. Every test passed, the suite was green, and the bug was
only found by calling the endpoint by hand months later. A route that cannot answer is not a
missing feature; it is a broken page.

So: enumerate the app's own OpenAPI schema (which is what the frontend and the docs read) and call
everything, substituting a plausible value for each path parameter. Anything 5xx, or any transport
failure, fails the test — with the list of offenders, not just the first one.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from apps.api.main import app, registry

#: Plausible values for path parameters. A 404 for an unknown id is a correct answer; a 500 is not.
PATH_VALUES = {
    "module_id": "churn",
    "session_id": "s_does_not_exist",
    "tool_name": "list_knowledge_sources",
    "name": "list_knowledge_sources",
}


@pytest.fixture(scope="module")
def client():
    if len(registry) == 0:
        registry.discover("modules")
    with TestClient(app) as c:
        yield c


def get_routes() -> list[str]:
    paths = app.openapi()["paths"]
    routes: list[str] = []
    for path, operations in paths.items():
        if "get" not in operations:
            continue
        concrete = path
        for param, value in PATH_VALUES.items():
            concrete = concrete.replace(f"{{{param}}}", value)
        if "{" in concrete:  # a path parameter this helper does not know about
            continue
        routes.append(concrete)
    return sorted(routes)


def test_the_schema_publishes_the_routes_the_frontend_calls(client) -> None:
    """A route the UI depends on must exist — this catches a rename that only breaks the browser."""
    routes = set(get_routes())
    for required in (
        "/health",
        "/api/v1/modules",
        "/api/v1/agent",
        "/api/v1/agent/tools",
        "/api/v1/integrations",
        "/api/v1/tenant",
        "/api/v1/warmup",
        "/api/v1/runs",
    ):
        assert required in routes, f"{required} is missing from the API"


def test_no_get_route_returns_a_server_error(client) -> None:
    broken: list[str] = []
    for path in get_routes():
        r = client.get(path)
        if r.status_code >= 500:
            broken.append(f"{path} -> {r.status_code} {r.text[:120]}")
    assert not broken, "routes that failed to answer:\n  " + "\n  ".join(broken)


def test_route_failures_say_why(client) -> None:
    """A 4xx from a route must carry a message — an empty detail is unactionable."""
    for path in get_routes():
        r = client.get(path)
        if 400 <= r.status_code < 500:
            detail = r.json().get("detail")
            assert detail, f"{path} returned {r.status_code} with an empty detail"


def test_the_agent_badge_endpoint_reports_a_provider(client) -> None:
    """The specific regression: this endpoint is what the chat header reads."""
    r = client.get("/api/v1/agent")
    assert r.status_code == 200
    body = r.json()
    assert body["provider"], "the badge needs a provider name"
    assert isinstance(body["live"], bool)
    assert body["tools"], "the header counts tools"
    # A mock must explain itself rather than looking like a broken product.
    if not body["live"]:
        assert body["note"] and "mock" in body["note"].lower()
