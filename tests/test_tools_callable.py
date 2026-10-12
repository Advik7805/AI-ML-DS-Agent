"""Every published tool must be runnable from the console with the arguments the UI prefills.

The Tool console builds its default arguments from each tool's JSON Schema (``examples`` first,
then ``default``, then a type-appropriate zero) and posts them. That makes the schema the contract
between the backend and a working "Run tool" button: a required property with no usable sample, or
an enum without a default, produces a 422 for a user who simply clicked the button.

This test mirrors the console's ``sampleValue`` exactly and calls all of them, so "can I try every
tool?" is answered by running them rather than by reading the schemas.
"""

from __future__ import annotations

import pytest

#: Mirrors `sampleValue` in apps/web/components/body/BodyExplorer.tsx — deliberately, including
#: its blind spots (no enum handling), because the point is to test what the UI actually sends.
def console_sample(schema: dict) -> object:
    if not isinstance(schema, dict):
        return None
    if "default" in schema:
        return schema["default"]
    if "examples" in schema:
        return schema["examples"][0]
    declared = schema.get("type")
    types = declared if isinstance(declared, list) else [declared]
    if "number" in types or "integer" in types:
        return 0
    if "string" in types:
        return ""
    return None


def console_arguments(tool: dict) -> dict:
    props = (tool.get("parameters_schema") or {}).get("properties") or {}
    return {name: console_sample(spec) for name, spec in props.items()}


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from apps.api.main import app, registry

    if len(registry) == 0:
        registry.discover("modules")
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module", params=["churn", "fraud", "maintenance", "pricing", "rag", "scraper", "xai_health"])
def module_tools(request, client):
    manifest = client.get(f"/api/v1/modules/{request.param}").json()
    return request.param, manifest["tools"]


def test_every_tool_answers_when_the_console_prefills_it(module_tools, client) -> None:
    body, tools = module_tools
    assert tools, f"{body} publishes no tools"
    broken: list[str] = []
    for tool in tools:
        args = console_arguments(tool)
        r = client.post(f"/api/v1/agent/tools/{tool['name']}", json={"arguments": args})
        if r.status_code != 200:
            broken.append(f"{tool['name']} -> {r.status_code} {r.text[:140]}")
    assert not broken, "tools that cannot be run from a prefilled console:\n  " + "\n  ".join(broken)


def test_no_published_tool_fails_with_an_empty_error(client) -> None:
    """A failing call must say why. `500 {"detail": ""}` tells the caller nothing."""
    tool_names = [t["name"] for t in client.get("/api/v1/agent/tools").json()["tools"]]
    assert tool_names, "no tools registered"
    for name in tool_names:
        r = client.post(f"/api/v1/agent/tools/{name}", json={"arguments": {}})
        if r.status_code >= 400:
            detail = r.json().get("detail")
            assert detail not in (None, ""), f"{name} failed with an empty message ({r.status_code})"
            if r.status_code >= 500:
                raise AssertionError(f"{name} raised a server error: {detail}")


def test_unknown_arguments_are_rejected_with_the_accepted_names(client) -> None:
    """A renamed argument must come back as a message naming the real ones, not a TypeError."""
    r = client.post(
        "/api/v1/agent/tools/get_churn_model_metrics",
        json={"arguments": {"nonsense": 1}},
    )
    assert r.status_code == 422
    assert "nonsense" in r.json()["detail"]
    assert "no arguments" in r.json()["detail"]


def test_missing_arguments_are_rejected_with_the_accepted_names(client) -> None:
    r = client.post("/api/v1/agent/tools/optimise_price", json={"arguments": {}})
    assert r.status_code == 422
    assert "sku" in r.json()["detail"]
    assert "accepts" in r.json()["detail"]


def test_a_string_instead_of_a_list_is_accepted_for_extract_url(client) -> None:
    r = client.post(
        "/api/v1/agent/tools/extract_url",
        json={"arguments": {"urls": "https://example.com"}},
    )
    assert r.status_code == 200, r.text[:200]
    result = r.json()["result"]
    assert result["results"], "a single URL must be extracted like a list of one"
    assert result["results"][0]["url"] == "https://example.com"
