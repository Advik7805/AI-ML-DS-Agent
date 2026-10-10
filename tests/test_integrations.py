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


# --- rate limiting -------------------------------------------------------------


def test_in_memory_limiter_enforces_the_limit() -> None:
    from core.integrations.ratelimit import InMemoryLimiter

    limiter = InMemoryLimiter(limit=3, window=60)
    outcomes = [limiter.check("client-a").allowed for _ in range(5)]
    assert outcomes == [True, True, True, False, False]


def test_in_memory_limiter_is_scoped_per_key() -> None:
    """One client tripping the limit must not lock out everyone else."""
    from core.integrations.ratelimit import InMemoryLimiter

    limiter = InMemoryLimiter(limit=1, window=60)
    assert limiter.check("a").allowed is True
    assert limiter.check("a").allowed is False
    assert limiter.check("b").allowed is True, "limits must be per client, not global"


def test_limiter_reports_its_own_weakness(monkeypatch) -> None:
    """An in-memory counter is per-instance and resets on restart. Say so, do not imply more."""
    from core.integrations import ratelimit

    for var in ("UPSTASH_REDIS_REST_URL", "UPSTASH_REDIS_REST_TOKEN", "NEXUS_UPSTASH_LIVE"):
        monkeypatch.delenv(var, raising=False)
    ratelimit.reset_limiter()
    s = ratelimit.status()
    assert s["live"] is False
    assert s["scope"] == "per process only"
    assert s["warning"], "the fallback must disclose that it is not a shared limit"


def test_upstash_is_selected_only_when_fully_configured(monkeypatch) -> None:
    """URL without a token, or a token without the flag, must not half-enable."""
    from core.integrations import ratelimit

    monkeypatch.setenv("UPSTASH_REDIS_REST_URL", "https://example.upstash.io")
    monkeypatch.setenv("UPSTASH_REDIS_REST_TOKEN", "tok")
    monkeypatch.delenv("NEXUS_UPSTASH_LIVE", raising=False)
    ratelimit.reset_limiter()
    assert ratelimit.get_limiter().backend == "in-memory"

    monkeypatch.setenv("NEXUS_UPSTASH_LIVE", "1")
    ratelimit.reset_limiter()
    assert ratelimit.get_limiter().backend == "upstash"
    ratelimit.reset_limiter()


def test_limiter_prunes_stale_keys() -> None:
    """A long-lived 512 MB instance must not accumulate a counter per client forever."""
    from core.integrations.ratelimit import InMemoryLimiter

    limiter = InMemoryLimiter(limit=100, window=0)  # window 0 => everything is immediately stale
    for i in range(5100):
        limiter.check(f"client-{i}")
    assert len(limiter._counts) < 5000


# --- gateway middleware --------------------------------------------------------


def test_health_is_exempt_from_rate_limiting() -> None:
    """The keep-alive cron hits /health every 10 minutes and must never trip the limit."""
    from fastapi.testclient import TestClient

    from apps.api.main import app

    with TestClient(app) as client:
        for _ in range(20):
            assert client.get("/health").status_code == 200


def test_metered_route_returns_429_with_honest_headers(monkeypatch) -> None:
    from fastapi.testclient import TestClient

    from apps.api.main import app
    from core.integrations import ratelimit

    # One shared instance, exactly as the memoised get_limiter() returns in production. Building
    # a fresh limiter per call would reset the counter on every request and prove nothing.
    limiter = ratelimit.InMemoryLimiter(limit=2, window=60)
    monkeypatch.setattr("apps.api.main.get_limiter", lambda: limiter)

    with TestClient(app) as client:
        codes = [client.get("/api/v1/modules").status_code for _ in range(4)]
        assert codes[:2] == [200, 200], codes
        assert codes[2] == 429, codes
        assert codes[3] == 429, codes
        r = client.get("/api/v1/modules")
        assert r.headers["X-RateLimit-Backend"] == "in-memory"
        assert r.headers["X-RateLimit-Limit"] == "2"


# --- Firecrawl -----------------------------------------------------------------


def test_firecrawl_defaults_to_a_labelled_mock() -> None:
    from core.integrations.firecrawl import MockFirecrawl, get_client

    client = get_client()
    assert isinstance(client, MockFirecrawl)
    out = client.scrape("https://example.com/app")
    assert out["backend"] == "mock"
    assert out["markdown"].startswith("[mock]")


def test_firecrawl_key_plus_flag_selects_the_live_client(monkeypatch) -> None:
    from core.integrations import config as cfg
    from core.integrations.firecrawl import FirecrawlClient, MockFirecrawl, get_client

    monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-x")
    monkeypatch.delenv("NEXUS_FIRECRAWL_LIVE", raising=False)
    assert cfg.FIRECRAWL.enabled is False, "a key alone must not enable a metered service"
    assert isinstance(get_client(), MockFirecrawl)

    monkeypatch.setenv("NEXUS_FIRECRAWL_LIVE", "1")
    assert isinstance(get_client(), FirecrawlClient)


def test_firecrawl_client_refuses_an_empty_key() -> None:
    from core.integrations.firecrawl import FirecrawlClient

    with pytest.raises(cfg.NotConfigured) as exc:
        FirecrawlClient("")
    assert "FIRECRAWL_API_KEY" in str(exc.value)


def test_render_url_is_wired_into_the_scraper_body(registry) -> None:
    scraper = registry.get("scraper")
    out = scraper.call_tool("render_url", {"url": "https://example.com/app"})
    assert out["backend"] == "mock"
    assert out["live"] is False
    assert out["markdown"].startswith("[mock]")


def test_render_url_validates_the_scheme(registry) -> None:
    with pytest.raises(ValueError, match="absolute http"):
        registry.get("scraper").call_tool("render_url", {"url": "example.com/no-scheme"})


def test_capabilities_reports_all_three_acquisition_paths(registry) -> None:
    """Plain GET, Tavily extract and Firecrawl render have different limits and different
    free-tier economics (Tavily refills monthly, Firecrawl does not). Conflating them would
    lead an operator to spend the wrong allowance."""
    caps = registry.get("scraper").call_tool("scraper_capabilities", {})
    assert "tavily" in caps and "firecrawl" in caps
    assert caps["tavily"]["live"] is False
    assert caps["firecrawl"]["live"] is False
    assert "one-time" in caps["firecrawl"]["free_tier"].lower()
