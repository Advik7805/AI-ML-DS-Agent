"""Firecrawl — JavaScript-rendering scrape, for pages a plain GET cannot read.

The scraper body already has two acquisition paths. This is the third, and the reason it exists
is narrow: a plain HTTP GET returns the pre-JavaDOM shell of a modern site, which for most
dashboards and docs portals is an empty `<div id="root">`. Tavily's extract covers the same
ground on a friendlier free tier, which is why this shipped second and stays opt-in.

Design notes, both from Firecrawl's own guidance:

- **Request the narrowest format that satisfies the consumer.** `markdown` only. Asking for
  markdown + html + rawHtml "just in case" multiplies the work and, on a metered plan, the bill.
- **Read the specific fields consumed, not the whole envelope.** `data.markdown` and
  `data.metadata.statusCode`. Metadata extras and error-envelope shapes differ between
  Firecrawl-compatible engines; field-level reads survive that, payload snapshots do not.

Free tier is 500 credits *one-time*, not monthly — so unlike Tavily this is not something to
leave switched on during development. That asymmetry is why the flag defaults off here too.
"""

from __future__ import annotations

from typing import Any, Protocol

from .config import FIRECRAWL, NotConfigured

API_BASE = "https://api.firecrawl.dev/v1"


class ScrapeBackend(Protocol):
    def scrape(self, url: str, **kwargs: Any) -> dict[str, Any]: ...


class MockFirecrawl:
    """Offline and deterministic, and it says so in the payload it returns."""

    backend = "mock"

    def scrape(self, url: str, **kwargs: Any) -> dict[str, Any]:
        return {
            "url": url,
            "backend": self.backend,
            "status": 200,
            "markdown": (
                f"[mock] Offline placeholder for the JavaScript-rendered content of {url}.\n\n"
                "Set FIRECRAWL_API_KEY and NEXUS_FIRECRAWL_LIVE=1 to fetch the real page. "
                "Until then nothing here was retrieved from the network, and none of it should "
                "be quoted or fed to another body as fact."
            ),
            "credits_used": 0,
            "note": "Simulated. The free tier is 500 credits one-time, not monthly.",
        }


class FirecrawlClient:
    """Real Firecrawl over HTTP. Raises rather than degrading, so misconfiguration is visible."""

    backend = "firecrawl"

    def __init__(self, api_key: str, timeout: float = 45.0) -> None:
        if not api_key:
            raise NotConfigured(FIRECRAWL)
        self.api_key = api_key
        self.timeout = timeout

    def scrape(self, url: str, **kwargs: Any) -> dict[str, Any]:
        import httpx

        payload: dict[str, Any] = {
            "url": url,
            # Narrowest format that satisfies the consumer. See module docstring.
            "formats": list(kwargs.get("formats", ["markdown"])),
        }
        if kwargs.get("only_main_content") is not None:
            payload["onlyMainContent"] = bool(kwargs["only_main_content"])
        if kwargs.get("wait_for"):
            payload["waitFor"] = int(kwargs["wait_for"])

        resp = httpx.post(
            f"{API_BASE}/scrape",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.timeout,
        )
        if resp.status_code == 401:
            raise NotConfigured(FIRECRAWL)
        if resp.status_code == 402:
            raise RuntimeError(
                "Firecrawl credits exhausted. The free tier is 500 one-time credits, so this "
                "does not refill monthly — fall back to Tavily extract or a plain GET."
            )
        if resp.status_code == 429:
            raise RuntimeError("Firecrawl rate limit reached; retry later.")
        resp.raise_for_status()

        body = resp.json()
        # Field-level reads, deliberately: the envelope shape varies between Firecrawl-compatible
        # engines and asserting on the whole payload is what breaks first.
        data = body.get("data") or {}
        metadata = data.get("metadata") or {}
        return {
            "url": url,
            "backend": self.backend,
            "status": int(metadata.get("statusCode") or 200),
            "markdown": data.get("markdown") or "",
            "title": metadata.get("title"),
            "description": metadata.get("description"),
        }


def get_client(force_mock: bool = False) -> ScrapeBackend:
    """Resolve the backend from the environment. Never raises."""
    if force_mock or not FIRECRAWL.enabled:
        return MockFirecrawl()
    try:
        return FirecrawlClient(FIRECRAWL.key)
    except NotConfigured:
        return MockFirecrawl()


def capabilities() -> dict[str, Any]:
    client = get_client()
    return {
        "backend": client.backend,
        "live": FIRECRAWL.enabled,
        "has_key": FIRECRAWL.has_key,
        "key_env": FIRECRAWL.key_env,
        "enable_env": FIRECRAWL.enable_env,
        "free_tier": FIRECRAWL.free_tier,
        "docs_url": FIRECRAWL.docs_url,
        "warning": None
        if FIRECRAWL.enabled
        else (
            "Running on deterministic mock data. Set "
            f"{FIRECRAWL.key_env} and {FIRECRAWL.enable_env}=1 to render real pages. Note the "
            "free tier is one-time, not monthly."
        ),
    }
