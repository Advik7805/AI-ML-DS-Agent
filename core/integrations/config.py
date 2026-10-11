"""External service configuration — the single place a deployment declares what it has.

The whole platform is built to run with **zero** keys. Every integration here is optional and
every consumer falls back to something honest, so a student can build and demo the entire
product on a laptop and then light up real backends one env var at a time.

Two rules this module enforces, both learned the hard way:

1. **Never silently substitute fake data for a real service.** A mock must announce itself in
   the payload it returns (``backend: "mock"``), because a RAG system that quietly answers from
   invented text is worse than one that says "I have nothing". This is the same failure the
   scraper body and the RAG grounding rule were written to prevent.
2. **Missing config is not an error.** ``configured`` is a property the UI reads, so a
   storefront can render "connect Tavily to enable live search" instead of throwing a 500.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Integration:
    """Declarative description of one optional external service."""

    id: str
    name: str
    purpose: str
    #: Env var holding the secret. Empty when the service needs no key.
    key_env: str = ""
    #: Env var that must be truthy to switch on live calls, even when a key exists.
    enable_env: str = ""
    #: Env vars that tune behaviour but do not gate it.
    option_env: tuple[str, ...] = field(default_factory=tuple)
    docs_url: str = ""
    free_tier: str = ""
    #: Which Saros bodies benefit. Purely informational, shown in the UI.
    used_by: tuple[str, ...] = field(default_factory=tuple)

    @property
    def key(self) -> str:
        return os.getenv(self.key_env, "").strip() if self.key_env else ""

    @property
    def has_key(self) -> bool:
        return bool(self.key)

    @property
    def enabled(self) -> bool:
        """True only when the service will actually be called.

        An explicit enable flag exists so a key can be present in a `.env` for local
        experiments without every deployment suddenly making billable calls.
        """
        if self.enable_env and not _flag(self.enable_env):
            return False
        if self.key_env:
            return self.has_key
        return _flag(self.enable_env) if self.enable_env else False

    def status(self) -> dict[str, object]:
        """Safe to serialise and send to the browser. Never includes the key itself."""
        options = {k: os.getenv(k, "") for k in self.option_env if os.getenv(k)}
        return {
            "id": self.id,
            "name": self.name,
            "purpose": self.purpose,
            "enabled": self.enabled,
            "has_key": self.has_key,
            "key_env": self.key_env,
            "enable_env": self.enable_env,
            "options": options,
            "docs_url": self.docs_url,
            "free_tier": self.free_tier,
            "used_by": list(self.used_by),
        }


class NotConfigured(RuntimeError):
    """Raised when live use is attempted without credentials.

    Callers catch this and degrade. It is deliberately loud rather than returning empty data,
    so a misconfigured deploy surfaces in the trace instead of looking like a bad result.
    """

    def __init__(self, integration: Integration) -> None:
        self.integration = integration
        hint = f"Set {integration.key_env}" if integration.key_env else f"Set {integration.enable_env}"
        if integration.key_env and integration.enable_env:
            hint += f" and {integration.enable_env}=1"
        super().__init__(
            f"{integration.name} is not configured. {hint}. "
            f"Until then this runs on deterministic mock data, which is labelled as such. "
            f"Free tier: {integration.free_tier or 'n/a'} — {integration.docs_url}"
        )


# --- the catalogue -------------------------------------------------------------

TAVILY = Integration(
    id="tavily",
    name="Tavily",
    purpose="Web search and page extraction for the agent's live research tool.",
    key_env="TAVILY_API_KEY",
    enable_env="SAROS_TAVILY_LIVE",
    option_env=("SAROS_TAVILY_MAX_RESULTS", "SAROS_TAVILY_SEARCH_DEPTH"),
    docs_url="https://docs.tavily.com",
    free_tier="1,000 credits/month, no credit card; students free. Basic search = 1 credit.",
    used_by=("scraper", "rag"),
)

FIRECRAWL = Integration(
    id="firecrawl",
    name="Firecrawl",
    purpose="Production-grade scraping that survives JavaScript-heavy pages.",
    key_env="FIRECRAWL_API_KEY",
    enable_env="SAROS_FIRECRAWL_LIVE",
    docs_url="https://docs.firecrawl.dev",
    free_tier="500 credits one-time on the free plan.",
    used_by=("scraper",),
)

GROQ = Integration(
    id="groq",
    name="Groq",
    purpose="Fast hosted LLM inference for the hub agent's planning step.",
    key_env="GROQ_API_KEY",
    docs_url="https://console.groq.com",
    free_tier="Free tier with per-minute token limits.",
    used_by=("agent",),
)

GEMINI = Integration(
    id="gemini",
    name="Google Gemini",
    purpose="Second LLM provider, so the agent is not tied to one vendor.",
    key_env="GEMINI_API_KEY",
    docs_url="https://aistudio.google.com",
    free_tier="Free tier with per-day request limits.",
    used_by=("agent",),
)

SUPABASE = Integration(
    id="supabase",
    name="Supabase",
    purpose="Postgres + pgvector for persisting RAG embeddings across cold starts.",
    key_env="SUPABASE_SERVICE_ROLE_KEY",
    enable_env="SAROS_SUPABASE_LIVE",
    option_env=("SUPABASE_URL",),
    docs_url="https://supabase.com/docs/guides/ai/vector-columns",
    free_tier="500 MB database, 2 projects, pauses after a week of inactivity.",
    used_by=("rag",),
)

UPSTASH = Integration(
    id="upstash",
    name="Upstash Redis",
    purpose="Rate limiting and caching, so a free-tier backend is not hammered.",
    key_env="UPSTASH_REDIS_REST_TOKEN",
    enable_env="SAROS_UPSTASH_LIVE",
    option_env=("UPSTASH_REDIS_REST_URL",),
    docs_url="https://upstash.com/docs/redis/overall/getstarted",
    free_tier="10,000 commands/day, 256 MB.",
    used_by=("api", "scraper"),
)

ALL: tuple[Integration, ...] = (TAVILY, FIRECRAWL, GROQ, GEMINI, SUPABASE, UPSTASH)

_BY_ID = {i.id: i for i in ALL}


def get(integration_id: str) -> Integration:
    try:
        return _BY_ID[integration_id]
    except KeyError:
        raise KeyError(f"unknown integration {integration_id!r}; known: {sorted(_BY_ID)}") from None


def statuses() -> list[dict[str, object]]:
    return [i.status() for i in ALL]


def summary() -> dict[str, object]:
    """One glance: what is live, what is mocked, what a deploy must set."""
    enabled = [i.id for i in ALL if i.enabled]
    return {
        "integrations": statuses(),
        "enabled": enabled,
        "missing": [
            {"id": i.id, "name": i.name, "key_env": i.key_env, "enable_env": i.enable_env}
            for i in ALL
            if not i.enabled
        ],
        "note": (
            "Everything runs without keys. Each integration degrades to a deterministic mock "
            "that labels itself, so nothing is silently faked."
        ),
    }
