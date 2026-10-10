"""Tavily — web search and page extraction for the agent.

Why Tavily: it is the only serious search API with a genuinely free student tier (1,000
credits/month, no card), and it returns cleaned, LLM-ready content rather than a list of
links to go fetch. Basic search costs 1 credit, extraction 1 credit per 5 URLs, so the free
allowance covers real development.

Three modes, chosen explicitly rather than guessed:

- **mock** (default) — deterministic, offline, and labelled. This is what CI and a fresh clone
  run on.
- **live** — real HTTP calls, gated behind ``NEXUS_TAVILY_LIVE=1`` *and* a key, so a key left
  in a local `.env` cannot make a demo accidentally billable.
- **error** — live requested but misconfigured; raises ``NotConfigured`` rather than quietly
  returning empty results.

The mock is honest about being a mock in every payload. An agent that cannot tell simulated
retrieval from real retrieval will confidently cite things that never happened.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Protocol

from .config import TAVILY, NotConfigured

API_BASE = "https://api.tavily.com"

#: A small, fixed corpus the mock "searches". Deterministic and clearly fictional, so nobody
#: mistakes a mock result for a real citation.
_MOCK_SOURCES = (
    (
        "Predictive maintenance benchmarks in 2026",
        "https://example.org/research/predictive-maintenance-benchmarks",
        "Fleet studies continue to report 20-40% reductions in unplanned downtime when "
        "remaining-useful-life estimates drive work-order scheduling, though most published "
        "figures come from vendor-sponsored trials on proprietary datasets.",
    ),
    (
        "Elasticity estimation from point-of-sale data",
        "https://example.org/research/price-elasticity-pos",
        "Own-price elasticity is rarely identifiable without deliberate price variation. "
        "Retailers running randomised price tests recover elasticity standard errors roughly "
        "a third of the size of those from purely observational history.",
    ),
    (
        "Fairness auditing under missing protected attributes",
        "https://example.org/research/fairness-proxy-variables",
        "When race and gender are unavailable, age band and geography are common proxies. "
        "Proxies understate disparity: a model can show equal error rates across proxy groups "
        "while still harming the protected population.",
    ),
    (
        "SMOTE and calibrated anomaly scoring",
        "https://example.org/research/smote-calibration",
        "Oversampling before calibration shifts the score distribution. Recalibrating on a "
        "held-out fold of the original, unbalanced data keeps predicted probabilities "
        "interpretable as real-world frequencies.",
    ),
)


class SearchBackend(Protocol):
    def search(self, query: str, **kwargs: Any) -> dict[str, Any]: ...

    def extract(self, urls: list[str], **kwargs: Any) -> dict[str, Any]: ...


def _seed(query: str) -> int:
    """Stable per-query seed, so the same question always returns the same mock results."""
    return int(hashlib.sha256(query.strip().lower().encode()).hexdigest()[:8], 16)


class MockTavily:
    """Offline, deterministic. Announces itself in every response."""

    backend = "mock"

    def search(self, query: str, **kwargs: Any) -> dict[str, Any]:
        k = int(kwargs.get("max_results", 5))
        seed = _seed(query)
        terms = [t for t in query.lower().split() if len(t) > 3][:4] or ["data"]

        results = []
        for offset in range(min(k, len(_MOCK_SOURCES))):
            title, url, content = _MOCK_SOURCES[(seed + offset) % len(_MOCK_SOURCES)]
            results.append(
                {
                    "title": title,
                    "url": url,
                    "content": content,
                    # Deterministic but query-dependent, so ranking still varies.
                    "score": round(0.95 - offset * 0.11 - (seed % 7) / 100, 4),
                    "published_date": "2026-01-15",
                }
            )
        return {
            "query": query,
            "backend": self.backend,
            "answer": (
                f"[mock] Offline placeholder for a Tavily search on {query!r}. "
                f"Set TAVILY_API_KEY and NEXUS_TAVILY_LIVE=1 for real results. "
                f"Matched terms: {', '.join(terms)}."
            ),
            "results": results,
            "credits_used": 0,
            "note": (
                "Simulated results. These URLs are illustrative placeholders and do not "
                "resolve — never cite them."
            ),
        }

    def extract(self, urls: list[str], **kwargs: Any) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "results": [
                {
                    "url": u,
                    "raw_content": (
                        f"[mock] Offline placeholder for the extracted text of {u}. "
                        "Set TAVILY_API_KEY and NEXUS_TAVILY_LIVE=1 to fetch the real page."
                    ),
                }
                for u in urls
            ],
            "failed_results": [],
            "credits_used": 0,
            "note": "Simulated extraction. No network call was made.",
        }


class TavilyClient:
    """Real Tavily over HTTP. Raises rather than degrading, so misconfiguration is visible."""

    backend = "tavily"

    def __init__(self, api_key: str, timeout: float = 30.0) -> None:
        if not api_key:
            raise NotConfigured(TAVILY)
        self.api_key = api_key
        self.timeout = timeout

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        import httpx

        resp = httpx.post(
            f"{API_BASE}{path}",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.timeout,
        )
        if resp.status_code == 401:
            raise NotConfigured(TAVILY)
        if resp.status_code == 429:
            raise RuntimeError(
                "Tavily rate limit or monthly credit allowance reached (free tier is 1,000 "
                "credits/month). Fall back to mock or wait for the reset on the first."
            )
        resp.raise_for_status()
        return resp.json()

    def search(self, query: str, **kwargs: Any) -> dict[str, Any]:
        payload = {
            "query": query,
            # `basic` is 1 credit; `advanced` is 2. Default to cheap — a student's 1,000
            # monthly credits disappear fast if every experiment costs double.
            "search_depth": kwargs.get("search_depth", "basic"),
            "max_results": int(kwargs.get("max_results", 5)),
            "include_answer": kwargs.get("include_answer", True),
        }
        for opt in ("topic", "time_range", "include_domains", "exclude_domains",
                    "include_raw_content"):
            if kwargs.get(opt) is not None:
                payload[opt] = kwargs[opt]
        out = self._post("/search", payload)
        out["backend"] = self.backend
        return out

    def extract(self, urls: list[str], **kwargs: Any) -> dict[str, Any]:
        # Billed per 5 successful extractions, so cap the batch by default.
        out = self._post(
            "/extract",
            {"urls": list(urls)[: int(kwargs.get("limit", 20))]},
        )
        out["backend"] = self.backend
        return out


def get_client(force_mock: bool = False) -> SearchBackend:
    """Resolve the backend from the environment. Never raises.

    The one place consumers ask for Tavily, so the mock/live decision cannot drift between
    the scraper body, the RAG body and the agent tool.
    """
    if force_mock or not TAVILY.enabled:
        return MockTavily()
    try:
        return TavilyClient(TAVILY.key)
    except NotConfigured:
        return MockTavily()


def capabilities() -> dict[str, Any]:
    """What the UI shows on the scraper and RAG storefronts."""
    client = get_client()
    import os

    return {
        "backend": client.backend,
        "live": TAVILY.enabled,
        "has_key": TAVILY.has_key,
        "key_env": TAVILY.key_env,
        "enable_env": TAVILY.enable_env,
        "max_results": int(os.getenv("NEXUS_TAVILY_MAX_RESULTS", "5") or 5),
        "search_depth": os.getenv("NEXUS_TAVILY_SEARCH_DEPTH", "basic") or "basic",
        "free_tier": TAVILY.free_tier,
        "docs_url": TAVILY.docs_url,
        "warning": None
        if TAVILY.enabled
        else (
            "Running on deterministic mock data. Results are placeholders and must not be "
            f"cited. Set {TAVILY.key_env} and {TAVILY.enable_env}=1 to go live."
        ),
    }
