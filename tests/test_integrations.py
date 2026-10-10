"""Integration layer — env gating, honest mocks, and no secret leakage.

Live HTTP calls to Tavily are deliberately *not* tested: the suite must not depend on a
third-party service being up, and a student's free credits should not be burned by CI. What is
tested is everything that can go wrong without a network — that the mock is deterministic and
labelled, that gating works, and that a status payload can never carry a key.
"""

from __future__ import annotations

import json

import pytest

from core.integrations import config as cfg
from core.integrations.tavily import MockTavily, TavilyClient, capabilities, get_client


def test_no_integration_is_enabled_by_default(monkeypatch) -> None:
    """A fresh clone with an empty environment must run entirely on mocks."""
    for i in cfg.ALL:
        for var in (i.key_env, i.enable_env, *i.option_env):
            if var:
                monkeypatch.delenv(var, raising=False)
    assert cfg.summary()["enabled"] == []
    assert len(cfg.summary()["missing"]) == len(cfg.ALL)


def test_status_never_leaks_a_secret(monkeypatch) -> None:
    """This endpoint is public. Reporting the env var *name* is fine; the value is not."""
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-SUPER-SECRET-VALUE")
    payload = json.dumps(cfg.summary())
    assert "tvly-SUPER-SECRET-VALUE" not in payload
    tavily = next(i for i in cfg.summary()["integrations"] if i["id"] == "tavily")
    assert tavily["key_env"] == "TAVILY_API_KEY"
    assert tavily["has_key"] is True


def test_a_key_alone_does_not_enable_live_calls(monkeypatch) -> None:
    """A key in a local .env must not make a demo billable.

    Gating needs both the key and the explicit flag.
    """
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-x")
    monkeypatch.delenv("NEXUS_TAVILY_LIVE", raising=False)
    assert cfg.TAVILY.has_key is True
    assert cfg.TAVILY.enabled is False
    assert get_client().backend == "mock"


def test_key_plus_flag_selects_the_live_client(monkeypatch) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-x")
    monkeypatch.setenv("NEXUS_TAVILY_LIVE", "1")
    assert cfg.TAVILY.enabled is True
    assert isinstance(get_client(), TavilyClient)


def test_live_client_refuses_an_empty_key() -> None:
    with pytest.raises(cfg.NotConfigured) as exc:
        TavilyClient("")
    # The message must tell the reader exactly which variables to set.
    assert "TAVILY_API_KEY" in str(exc.value)
    assert "NEXUS_TAVILY_LIVE" in str(exc.value)


def test_mock_search_is_deterministic() -> None:
    """Anything asserted on in a test must be stable across runs."""
    a = MockTavily().search("price elasticity in retail")
    b = MockTavily().search("price elasticity in retail")
    assert a == b
    assert a["results"], "an empty mock makes every downstream test vacuous"


def test_mock_search_announces_itself() -> None:
    """The central rule: a mock must never be mistaken for real retrieval."""
    out = MockTavily().search("anything")
    assert out["backend"] == "mock"
    assert "mock" in out["answer"].lower()
    assert "must not" in out["note"].lower() or "placeholders" in out["note"].lower()


def test_mock_extract_announces_itself() -> None:
    out = MockTavily().extract(["https://example.com"])
    assert out["backend"] == "mock"
    assert out["results"][0]["raw_content"].startswith("[mock]")


def test_capabilities_reports_what_a_deploy_must_set(monkeypatch) -> None:
    for var in ("TAVILY_API_KEY", "NEXUS_TAVILY_LIVE"):
        monkeypatch.delenv(var, raising=False)
    caps = capabilities()
    assert caps["live"] is False
    assert caps["key_env"] == "TAVILY_API_KEY"
    assert caps["warning"], "an unconfigured deployment must carry an explicit warning"


def test_scraper_tools_label_mock_output(registry) -> None:
    """The agent-facing surface must say so, not just the client."""
    scraper = registry.get("scraper")
    search = scraper.call_tool("web_search", {"query": "churn benchmarks", "max_results": 3})
    assert search["backend"] == "mock"
    assert search["live"] is False

    extract = scraper.call_tool("extract_url", {"urls": ["https://example.com"]})
    assert extract["backend"] == "mock"
    assert extract["urls"] == ["https://example.com"]


def test_extract_url_validates_urls(registry) -> None:
    scraper = registry.get("scraper")
    with pytest.raises(ValueError, match="absolute http"):
        scraper.call_tool("extract_url", {"urls": ["example.com/no-scheme"]})
    with pytest.raises(ValueError, match="Missing required"):
        scraper.call_tool("extract_url", {"urls": []})


def test_capabilities_reports_both_acquisition_paths(registry) -> None:
    """Plain HTTP and Tavily have different limits; conflating them would mislead."""
    caps = registry.get("scraper").call_tool("scraper_capabilities", {})
    assert "tavily" in caps
    assert caps["tavily"]["live"] is False
    assert "max_bytes" in caps


def test_unknown_integration_is_loud() -> None:
    with pytest.raises(KeyError, match="known:"):
        cfg.get("does_not_exist")
