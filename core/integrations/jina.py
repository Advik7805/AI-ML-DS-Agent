"""Jina Reader — JS-rendering scrape with no API key at all.

This is the default backend for `render_url`, and the reason is economic rather than technical.

Firecrawl renders pages well but gives 500 credits *once* — they do not refill, so a demo that
works on Tuesday is dead by Thursday. Jina Reader's anonymous endpoint costs nothing and has no
key to lose: prefix a URL with `https://r.jina.ai/` and markdown comes back. The anonymous tier
is rate limited to ~20 requests/minute; a free key (10M tokens, no card) raises that to 500 RPM.

So the order is: Jina by default because it always works, Firecrawl only when explicitly
configured, because spending a one-time allowance on something that is free elsewhere is a bad
trade.

Response shape: with `Accept: application/json` and `X-Respond-With: markdown` the body is
`{"code": 200, "status": 20000, "data": {"title", "url", "content", ...}}`. Plain text mode
returns a `Title:` / `URL Source:` header block followed by `Markdown Content:`. The JSON path
is used here because reading `data.content` is less brittle than parsing that header block.
"""

from __future__ import annotations

import os
from typing import Any

from .config import Integration

#: Declared here rather than in config.ALL because Jina works with no key, so it is not part of
#: the "what must a deploy configure" story that summary() tells.
JINA = Integration(
    id="jina",
    name="Jina Reader",
    purpose="JS-rendering scrape, no API key required",
    key_env="JINA_API_KEY",
    enable_env="",
    docs_url="https://jina.ai/reader",
    free_tier="~20 req/min anonymous; 500 RPM and 10M tokens with a free key",
    used_by=["scraper"],
)

API_BASE = "https://r.jina.ai/"


class JinaReader:
    """Fetches a URL through Jina Reader and returns markdown.

    A key is optional. With one, throughput goes from ~20 RPM to 500 RPM; without one the
    endpoint still works, which is the entire reason this is the default.
    """

    def __init__(self, api_key: str = "", timeout: float = 25.0) -> None:
        self.api_key = api_key.strip()
        self.timeout = timeout

    @property
    def backend(self) -> str:
        return "jina" if self.api_key else "jina-anonymous"

    @property
    def live(self) -> bool:
        return True

    def scrape(self, url: str, wait_for: int = 0) -> dict[str, Any]:
        import httpx

        headers = {
            "Accept": "application/json",
            "X-Respond-With": "markdown",
            # Reader's anonymous tier is the shared, polite one. Identify the caller.
            "User-Agent": "saros-agent/0.1 (+https://github.com/Advik7805/AI-ML-DS-Agent)",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if wait_for:
            # Selector to wait for before snapshotting; Jina's own header.
            headers["X-Wait-For-Selector"] = str(wait_for) if not str(wait_for).startswith(("#", ".")) else str(wait_for)

        resp = httpx.get(API_BASE + url, headers=headers, timeout=self.timeout)

        if resp.status_code == 429:
            raise RuntimeError(
                "Jina Reader rate limit hit (anonymous tier is ~20 req/min). Set JINA_API_KEY "
                "for 500 RPM, or retry shortly."
            )
        if resp.status_code == 401:
            raise RuntimeError("Jina Reader rejected the API key.")
        resp.raise_for_status()

        try:
            body = resp.json()
        except ValueError:
            # Plain-text mode fallback: strip the header block Jina prepends.
            text = resp.text
            marker = "Markdown Content:"
            return {
                "url": url,
                "title": "",
                "status": resp.status_code,
                "markdown": text.split(marker, 1)[-1].strip() if marker in text else text,
                "backend": self.backend,
            }

        data = body.get("data") or {}
        return {
            "url": data.get("url") or url,
            "title": data.get("title") or "",
            "status": data.get("statusCode") or body.get("status") or resp.status_code,
            "markdown": data.get("content") or "",
            "backend": self.backend,
        }


_client: JinaReader | None = None


def get_client(force_reset: bool = False) -> JinaReader:
    """Resolved once per process. Never raises: no key is a valid configuration."""
    global _client
    if _client is None or force_reset:
        _client = JinaReader(os.getenv("JINA_API_KEY", ""))
    return _client


def reset_client() -> None:
    global _client
    _client = None


def capabilities() -> dict[str, Any]:
    client = get_client()
    return {
        "backend": client.backend,
        "live": client.live,
        "keyless": not client.api_key,
        "rate_limit": "500 req/min with a key" if client.api_key else "~20 req/min anonymous",
        "note": (
            "Rendering through Jina Reader with an API key."
            if client.api_key
            else (
                "Rendering through Jina Reader's anonymous endpoint - free, no key, ~20 req/min. "
                "Set JINA_API_KEY for 500 RPM."
            )
        ),
    }
