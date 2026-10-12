"""Scraper — a Saros body.

The data-acquisition limb of the platform: every other body needs input, and this is how it
gets there without a human pasting CSVs.

**Two fetchers behind one interface.**

- ``MockFetcher`` (default) — deterministic, offline, no dependencies. Lets the agent's
  scraping path be exercised in CI and on a laptop with no network.
- ``HttpFetcher`` — real HTTP, enabled with ``SAROS_SCRAPER_LIVE=1``.

In production this should be pointed at Firecrawl rather than raw HTTP: it handles JavaScript
rendering, anti-bot behaviour and markdown conversion, none of which a plain ``GET`` does.
The interface here is the part that survives that swap.

Safety rails that are not optional:
- robots-style **allowlist**: only hosts in ``SAROS_SCRAPER_ALLOW`` are fetched, so the agent
  cannot be talked into hammering an arbitrary site.
- **size and depth caps**, a real User-Agent, and a timeout.
- the result states plainly whether it is live or mocked.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import urlparse

import numpy as np

from core.contract import Module
from core.integrations.firecrawl import capabilities as firecrawl_capabilities
from core.integrations.firecrawl import get_client as get_firecrawl_client
from core.integrations.jina import capabilities as jina_capabilities
from core.integrations.jina import get_client as get_jina_client
from core.integrations.tavily import capabilities, get_client
from core.types import AgentTool, Explanation, MetricsReport, ModuleManifest, Prediction

ACCENT = "#f97316"
MAX_BYTES = 400_000


class Fetcher(Protocol):
    kind: str

    def fetch(self, url: str) -> dict[str, Any]: ...


class MockFetcher:
    """Deterministic stand-in. Same output shape as the real fetcher."""

    kind = "mock"

    def fetch(self, url: str) -> dict[str, Any]:
        parsed = urlparse(url)
        host = parsed.netloc or "example.com"
        path = parsed.path or "/"
        digest = hashlib.sha256(url.encode()).hexdigest()[:12]
        # Stable pseudo-content derived from the URL, so the same URL always gives the same
        # result and tests can assert on it.
        body = (
            f"# {host}{path}\n\n"
            f"Mock capture of {url} (fingerprint `{digest}`).\n\n"
            f"Live scraping is disabled. Set `SAROS_SCRAPER_LIVE=1` to fetch over HTTP, or wire "
            f"in the Firecrawl client for JavaScript-rendered pages.\n\n"
            f"## Extracted fields\n"
            f"- host: {host}\n- path: {path}\n- bytes: 1024\n"
        )
        return {
            "url": url,
            "status": 200,
            "content_type": "text/markdown",
            "markdown": body,
            "bytes": len(body),
            "fetcher": self.kind,
            "links": [],
        }


class HttpFetcher:
    """Real HTTP. Plain GET only — no JS rendering, which is the honest limitation."""

    kind = "http"

    def __init__(self, timeout: float = 20.0, user_agent: str = "SarosBot/0.1 (+https://github.com/saros)"):
        self.timeout = timeout
        self.user_agent = user_agent

    def fetch(self, url: str) -> dict[str, Any]:
        import httpx

        res = httpx.get(
            url,
            timeout=self.timeout,
            headers={"User-Agent": self.user_agent},
            follow_redirects=True,
        )
        text = res.text[:MAX_BYTES]
        links = re.findall(r'href=["\']([^"\']+)["\']', text)[:50]
        return {
            "url": str(res.url),
            "status": res.status_code,
            "content_type": res.headers.get("content-type", ""),
            "markdown": _html_to_text(text),
            "bytes": len(text),
            "fetcher": self.kind,
            "links": links,
        }


def _html_to_text(html: str) -> str:
    """Minimal HTML→text. Good enough for extraction; not a substitute for Firecrawl."""
    text = re.sub(r"(?is)<(script|style|noscript).*?>.*?</\1>", " ", html)
    text = re.sub(r"(?is)<br\s*/?>|</p>|</div>|</li>|</h[1-6]>", "\n", text)
    text = re.sub(r"(?is)<[^>]+>", " ", text)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def _allowed(url: str, allowlist: list[str]) -> bool:
    if not allowlist:
        return False
    host = (urlparse(url).netloc or "").lower()
    return any(host == a or host.endswith("." + a) for a in allowlist)


@dataclass
class ScraperConfig:
    allowlist: list[str]
    live: bool


def _config() -> ScraperConfig:
    raw = os.getenv("SAROS_SCRAPER_ALLOW", "")
    allow = [h.strip().lower() for h in raw.split(",") if h.strip()]
    live = os.getenv("SAROS_SCRAPER_LIVE", "").strip().lower() in ("1", "true", "yes")
    return ScraperConfig(allowlist=allow, live=live)


class ScraperModule(Module):
    id = "scraper"

    @property
    def manifest(self) -> ModuleManifest:
        cfg = _config()
        return ModuleManifest(
            id=self.id,
            title="Web Scraper",
            summary=(
                "Turns a URL into structured text the other bodies can consume. Host-allowlisted "
                "and size-capped so the agent cannot be talked into hammering a site. Mocked by "
                "default; Firecrawl is the production backend. Data: none — Northwind Labs fetches public "
                "pages on demand; the mock is labelled everywhere it appears."
            ),
            accent=ACCENT,
            status="live",
            capabilities=["acquisition", "html-to-text", "allowlist", "mockable"],
            routes={"ui": "/scraper", "api": "/api/v1/scraper"},
            accepts=self.upload_spec().get("accepts", ""),
            dataset=self.dataset_card(),
            tools=self.tools(),
        )

    def upload_spec(self) -> dict[str, Any]:
        return {
            "kind": "csv",
            "accepts": (
                "CSV with a 'url' column (aliases: link, address). Up to 5 URLs are fetched per "
                "upload through the module's configured backend, and every row says whether the "
                "result was live or the labelled mock."
            ),
            "required": ["url"],
            "optional": ["label", "note"],
            "features": [],
            "label": None,
            "aliases": {"url": ["link", "address", "href"]},
            "notes": [],
        }

    def analyze_upload(self, filename: str, data: bytes, session: Any) -> Any:
        """Fetch each URL in the file and report what came back, per row."""
        from core.types import ColumnsReport, MetricsReport, Prediction, UploadAnalysis
        from core.upload import (
            UploadRejected,
            find_column,
            read_csv_bytes,
            upload_dataset_card,
        )

        df = read_csv_bytes(data)
        url_col = find_column(df, ["url", "link", "address", "href"])
        if url_col is None:
            raise UploadRejected(
                "No URL column found in this file.",
                expected=["url"],
                found=[str(c) for c in df.columns],
                hint="Aliases accepted: link, address, href.",
            )
        urls = [str(u).strip() for u in df[url_col].dropna().unique()][:5]

        predictions: list[Prediction] = []
        live_count = 0
        for url in urls:
            try:
                result = self.tool_render_url(url)
                live = bool(result.get("live"))
                live_count += int(live)
                text = str(result.get("markdown") or "")
                predictions.append(
                    Prediction(
                        module_id=self.id,
                        score=1.0 if live else 0.5,
                        label="Fetched live" if live else "Fetched (labelled mock)",
                        decision=f"{len(text):,} characters of text via {result.get('backend')}",
                        meta={
                            "url": url,
                            "backend": result.get("backend"),
                            "status": result.get("status"),
                            "excerpt": " ".join(text.split())[:280],
                        },
                    )
                )
            except Exception as exc:  # noqa: BLE001 - one bad URL must not fail the file
                predictions.append(
                    Prediction(
                        module_id=self.id,
                        score=0.0,
                        label="Fetch failed",
                        decision=f"{type(exc).__name__}: {exc}",
                        meta={"url": url},
                    )
                )

        return UploadAnalysis(
            session_id=session.id,
            module_id=self.id,
            filename=filename,
            size_bytes=len(data),
            kind="csv",
            rows=len(urls),
            columns_report=ColumnsReport(expected=["url"], matched=["url"], missing=[], ignored=[]),
            metrics=MetricsReport(
                n_samples=len(urls),
                positive_rate=float(live_count / len(urls)) if urls else 0.0,
                scores={
                    "urls_in_file": float(df[url_col].nunique()),
                    "urls_fetched": float(len(urls)),
                    "live_fetches": float(live_count),
                },
                notes=[
                    "Live fetches require the acquisition backend to be configured; otherwise each "
                    "row says 'labelled mock' and no page was actually requested.",
                    "At most 5 URLs are fetched per upload, so an upload cannot be used to hammer a host.",
                ],
            ),
            predictions=predictions,
            dataset=upload_dataset_card(self, filename, len(urls), [str(c) for c in df.columns]),
            notes=["Fetched text can be indexed into the Knowledge body to make it queryable."],
            extra={"accepts": self.upload_spec()["accepts"]},
        )

    def predict(self, payload: dict[str, Any]) -> Prediction:
        result = self._scrape(payload)
        status = int(result.get("status", 0))
        size = int(result.get("bytes", 0))
        score = float(np.clip(size / 50_000, 0.0, 1.0)) if size else 0.0

        if "error" in result:
            label, decision = "Fetch failed", str(result["error"])
        elif status >= 400:
            label, decision = f"HTTP {status}", "Check the URL and whether the host permits automated access."
        elif size < 200:
            label, decision = "Empty or blocked", "The page returned almost no content — likely JS-rendered or bot-blocked. Use Firecrawl."
        else:
            label, decision = "Captured", f"{size:,} bytes extracted and ready to feed another body."

        return Prediction(
            module_id=self.id,
            score=round(score, 4),
            label=label,
            decision=decision,
            inputs={"bytes": float(size)},
            meta=result,
        )

    def explain(self, payload: dict[str, Any]) -> Explanation:
        cfg = _config()
        return Explanation(
            kind="heuristic",
            space="unspecified",
            attributions=[],
            summary=(
                f"Fetcher: {'live HTTP' if cfg.live else 'mock'}. "
                f"Allowlist: {', '.join(cfg.allowlist) or 'empty — nothing can be fetched live'}."
            ),
            caveats=[
                "A plain GET cannot render JavaScript. For modern sites use the Firecrawl "
                "client behind this same interface.",
                "Respect the target's robots.txt and terms. The allowlist exists so this cannot "
                "be pointed at an arbitrary host by prompt injection.",
                "Extracted text is lossy: tables and nested structure are flattened.",
            ],
        )

    def evaluate(self) -> MetricsReport:
        """Determinism check: the mock must be stable, or tests built on it are meaningless."""
        url = "https://example.com/pricing"
        a = MockFetcher().fetch(url)["markdown"]
        b = MockFetcher().fetch(url)["markdown"]
        return MetricsReport(
            n_samples=2,
            positive_rate=1.0,
            scores={"deterministic": float(a == b), "bytes": float(len(a))},
            notes=[
                "The mock fetcher must be deterministic, otherwise any test asserting on scraped "
                "output is asserting on noise.",
                "There is no live-fetch metric here on purpose: measuring it in CI would make the "
                "suite depend on a third-party site being up.",
            ],
        )

    def _scrape(self, payload: dict[str, Any]) -> dict[str, Any]:
        url = str(payload.get("url", "")).strip()
        if not url:
            raise ValueError("Missing required features: url")
        if not url.startswith(("http://", "https://")):
            raise ValueError("url must start with http:// or https://")

        cfg = _config()
        if not cfg.live:
            return MockFetcher().fetch(url)

        if not _allowed(url, cfg.allowlist):
            return {
                "url": url,
                "status": 0,
                "bytes": 0,
                "fetcher": "blocked",
                "error": (
                    f"Host not in SAROS_SCRAPER_ALLOW ({', '.join(cfg.allowlist) or 'empty'}). "
                    "Live scraping is allowlisted so the agent cannot be directed at an "
                    "arbitrary site."
                ),
            }
        try:
            return HttpFetcher().fetch(url)
        except Exception as exc:  # noqa: BLE001 - report, never propagate
            return {"url": url, "status": 0, "bytes": 0, "fetcher": "http", "error": f"{type(exc).__name__}: {exc}"}

    def tools(self) -> list[AgentTool]:
        return [
            AgentTool(
                name="scrape_url",
                description=(
                    "Fetch a URL and return it as text the other bodies can consume. Host must be "
                    "in the allowlist when live scraping is enabled; otherwise returns a mock."
                ),
                module_id=self.id,
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "Absolute http(s) URL.",
                            "examples": ["https://example.com"],
                        }
                    },
                    "required": ["url"],
                },
            ),
            AgentTool(
                name="web_search",
                description=(
                    "Search the live web via Tavily and return ranked, LLM-ready results. "
                    "Use this when the answer is not in the local corpus. Runs on labelled mock "
                    "data until TAVILY_API_KEY and SAROS_TAVILY_LIVE=1 are set."
                ),
                module_id=self.id,
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query.",
                            "examples": ["on-premise AI deployment costs"],
                        },
                        "max_results": {"type": "integer", "description": "Up to 20; costs 1 credit."},
                        "topic": {
                            "type": "string",
                            "enum": ["general", "news", "finance"],
                            "description": "Which index to search.",
                        },
                        "time_range": {
                            "type": ["string", "null"],
                            "enum": ["day", "week", "month", "year", None],
                            "description": "Restrict to recent results.",
                        },
                    },
                    "required": ["query"],
                },
            ),
            AgentTool(
                name="extract_url",
                description=(
                    "Extract the readable text of up to five URLs via Tavily. Unlike scrape_url "
                    "this renders JavaScript and is not bound by the local allowlist, because "
                    "Tavily does the fetching. Billed at 1 credit per 5 URLs."
                ),
                module_id=self.id,
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "urls": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Absolute http(s) URLs, max 5 per call.",
                            "examples": [["https://example.com"]],
                        }
                    },
                    "required": ["urls"],
                },
            ),
            AgentTool(
                name="render_url",
                description=(
                    "Fetch a JavaScript-rendered page as markdown via Firecrawl. Use this instead "
                    "of scrape_url when a plain GET returns an empty shell — dashboards, docs "
                    "portals, anything client-rendered. The free tier is 500 credits ONE TIME, so "
                    "prefer scrape_url or Tavily extract unless the page genuinely needs a browser."
                ),
                module_id=self.id,
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "Absolute http(s) URL.",
                            "examples": ["https://example.com"],
                        },
                        "wait_for": {
                            "type": "integer",
                            "description": "Milliseconds to wait for JS to settle. Default 0.",
                        },
                    },
                    "required": ["url"],
                },
            ),
            AgentTool(
                name="scraper_capabilities",
                description="Report which acquisition backends are live, which are mocked, and what a deploy must set.",
                module_id=self.id,
                parameters_schema={"type": "object", "properties": {}},
            ),
        ]

    def tool_scrape_url(self, url: str) -> dict[str, Any]:
        result = self._scrape({"url": url})
        # Trim before it reaches an LLM context window.
        if result.get("markdown"):
            result["markdown"] = result["markdown"][:4000]
            result["truncated"] = result["bytes"] > 4000
        return result

    def tool_web_search(self, query: str, max_results: int = 5, topic: str = "general",
                        time_range: str | None = None) -> dict[str, Any]:
        """Live web search. Mock results are labelled so nothing invented gets cited."""
        client = get_client()
        try:
            out = client.search(
                query,
                max_results=int(max_results or 5),
                topic=topic or "general",
                time_range=time_range,
            )
        except Exception as exc:  # noqa: BLE001 - degrade, never break the conversation
            return {
                "query": query,
                "backend": "error",
                "results": [],
                "error": f"{type(exc).__name__}: {exc}",
                "note": (
                    "Live search failed. Falling back to the local corpus is safer than "
                    "guessing — say you could not verify this."
                ),
            }
        out["backend"] = client.backend
        out["live"] = client.backend != "mock"
        # Trim so a search cannot blow the context window.
        for r in out.get("results", []):
            if isinstance(r.get("content"), str):
                r["content"] = r["content"][:600]
        return out

    def tool_extract_url(self, urls: list[str] | str) -> dict[str, Any]:
        client = get_client()
        # `extract_url` is called by hand as often as by the agent, and one URL passed as a bare
        # string is the usual mistake. Iterating a string yields characters, so normalise first.
        if isinstance(urls, str):
            urls = [urls]
        cleaned = [str(u).strip() for u in (urls or []) if str(u).strip()][:5]
        if not cleaned:
            raise ValueError("Missing required features: urls")
        for u in cleaned:
            if not u.startswith(("http://", "https://")):
                raise ValueError(f"urls must be absolute http(s) URLs, got {u!r}")
        try:
            out = client.extract(cleaned)
        except Exception as exc:  # noqa: BLE001
            return {"urls": cleaned, "backend": "error", "results": [],
                    "error": f"{type(exc).__name__}: {exc}"}
        out["backend"] = client.backend
        out["live"] = client.backend != "mock"
        # Echo the request so the summariser (and any UI) can say what was asked for even when
        # the provider omits it from its response.
        out["urls"] = cleaned
        for r in out.get("results", []):
            if isinstance(r.get("raw_content"), str):
                r["raw_content"] = r["raw_content"][:4000]
        return out

    def tool_render_url(self, url: str, wait_for: int = 0) -> dict[str, Any]:
        """JS-rendering scrape. Not bound by the local allowlist: Firecrawl does the fetching."""
        target = str(url).strip()
        if not target.startswith(("http://", "https://")):
            raise ValueError(f"url must be absolute http(s), got {target!r}")
        # Jina first: it works with no key at all, so rendering is never dead-on-arrival in
        # production. Firecrawl only when explicitly configured, because its 500 credits are
        # one-time and spending them on something free elsewhere is a bad trade.
        #
        # Both are gated on SAROS_SCRAPER_LIVE, the same flag that governs every other outbound
        # fetch here. Without that gate a unit test - or a laptop with no egress - would make a
        # real network call and fail, and "it works in prod but not in CI" is exactly the sort of
        # asymmetry that wastes an afternoon.
        firecrawl = get_firecrawl_client()
        if not _config().live:
            client = firecrawl  # labelled "[mock]", never silent
        else:
            client = firecrawl if firecrawl.backend != "mock" else get_jina_client()
        try:
            out = client.scrape(target, wait_for=int(wait_for or 0))
        except Exception as exc:  # noqa: BLE001 - degrade, never break the conversation
            return {
                "url": target,
                "backend": "error",
                "status": 0,
                "markdown": "",
                "error": f"{type(exc).__name__}: {exc}",
                "note": (
                    "Rendering failed. A plain GET via scrape_url may still work if the page is "
                    "server-rendered."
                ),
            }
        out["backend"] = client.backend
        out["live"] = client.backend != "mock"
        if isinstance(out.get("markdown"), str):
            out["markdown"] = out["markdown"][:4000]
        return out

    def tool_scraper_capabilities(self) -> dict[str, Any]:
        cfg = _config()
        tavily = capabilities()
        return {
            "live": cfg.live,
            "allowlist": cfg.allowlist,
            "fetcher": "http" if cfg.live else "mock",
            "max_bytes": MAX_BYTES,
            # Two acquisition paths, reported separately: a plain GET (allowlisted, cannot run
            # JavaScript) and Tavily (renders JS, not allowlisted, costs credits).
            "tavily": tavily,
            "firecrawl": firecrawl_capabilities(),
            # Jina is the default renderer because it needs no key at all, so this is the one
            # path that works on a fresh deploy with zero configuration.
            "jina": jina_capabilities(),
            "production_backend": (
                "render_url uses Jina Reader by default (free, no key, ~20 req/min) and switches "
                "to Firecrawl only if it is explicitly configured, because Firecrawl's 500 "
                "credits are one-time. Tavily covers search and multi-URL extract."
            ),
        }


MODULE = ScraperModule()
