"""Supabase — a corpus the knowledge base can grow without a redeploy.

The RAG body ships six markdown documents in `modules/rag/corpus/`. That is the right default:
they are versioned, reviewable, and work with no network. But it also means every new policy
document requires a commit, a CI run and a redeploy — which is a poor fit for something whose
whole purpose is answering questions from an evolving set of rules.

This adds an *optional* second source: rows in a Supabase table, fetched at retriever build time
and merged with the in-repo documents. The in-repo corpus always loads; the database only adds
to it. So a Supabase outage degrades to "the six documents I shipped with", never to "no
knowledge base".

What this deliberately is not: a vector store. At 25 passages, pgvector would be a network round
trip that buys nothing over TF-IDF, and it would add an embedding dependency to a body that
currently has none. The table stores plain text and the existing TF-IDF ranking still runs in
process. Revisit if the corpus grows past a few hundred passages.

Schema:

    create table knowledge_documents (
      id         bigint generated always as identity primary key,
      source     text not null,           -- slug, becomes the citation label
      heading    text not null default '',
      body       text not null,
      active     boolean not null default true,
      updated_at timestamptz not null default now()
    );
    alter table knowledge_documents enable row level security;
    -- RLS on with no policies: the service_role key used here bypasses it, and the anon key
    -- gets no access at all. That is the intended configuration.
"""

from __future__ import annotations

import os
from typing import Any, Protocol

from .config import SUPABASE, NotConfigured

TABLE = os.getenv("NEXUS_SUPABASE_TABLE", "knowledge_documents")


class DocumentSource(Protocol):
    def fetch_documents(self) -> list[dict[str, str]]: ...


class EmptySource:
    """The default. Contributes nothing; the in-repo corpus stands alone."""

    backend = "none"

    def fetch_documents(self) -> list[dict[str, str]]:
        return []


class SupabaseSource:
    """Reads additional corpus documents from PostgREST. Raises rather than degrading.

    Both `apikey` and `Authorization` are required — PostgREST rejects a request missing either,
    and the error it returns is unhelpful enough to be worth avoiding.
    """

    backend = "supabase"

    def __init__(self, url: str, key: str, table: str = TABLE, timeout: float = 10.0) -> None:
        if not url or not key:
            raise NotConfigured(SUPABASE)
        self.url = url.rstrip("/")
        self.key = key
        self.table = table
        self.timeout = timeout

    def fetch_documents(self) -> list[dict[str, str]]:
        import httpx

        resp = httpx.get(
            # Only the fields the retriever needs, and only rows that are switched on.
            f"{self.url}/rest/v1/{self.table}",
            headers={
                "apikey": self.key,
                "Authorization": f"Bearer {self.key}",
                "Accept": "application/json",
            },
            params={"select": "source,heading,body", "active": "eq.true", "order": "id.asc"},
            timeout=self.timeout,
        )
        if resp.status_code == 401:
            raise NotConfigured(SUPABASE)
        if resp.status_code == 404:
            raise RuntimeError(
                f"Table {self.table!r} not found in this Supabase project. Create it with the "
                "schema in this module's docstring, or unset NEXUS_SUPABASE_LIVE."
            )
        resp.raise_for_status()

        rows = resp.json()
        out = []
        for r in rows if isinstance(rows, list) else []:
            body = (r.get("body") or "").strip()
            source = (r.get("source") or "").strip()
            # A row with no text is worse than no row: it would become a passage that can never
            # be matched but still inflates the corpus size.
            if not body or not source:
                continue
            out.append({"source": source, "heading": (r.get("heading") or "").strip(), "body": body})
        return out


_source: object | None = None


def get_source(force_empty: bool = False) -> DocumentSource:
    """Resolve once per process. Never raises — a knowledge base must still work offline."""
    global _source
    if _source is None:
        url = os.getenv("SUPABASE_URL", "").strip()
        if force_empty or not SUPABASE.enabled or not url:
            _source = EmptySource()
        else:
            try:
                _source = SupabaseSource(url, SUPABASE.key)
            except NotConfigured:
                _source = EmptySource()
    return _source


def reset_source() -> None:
    """For tests, and for anything that changes the env at runtime."""
    global _source
    _source = None


def status() -> dict[str, Any]:
    source = get_source()
    live = getattr(source, "backend", "none") == "supabase"
    return {
        "backend": getattr(source, "backend", "none"),
        "live": live,
        "table": TABLE,
        "purpose": "extra corpus documents, merged with the in-repo markdown",
        "warning": None
        if live
        else (
            "Using only the in-repo corpus. Set SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY and "
            "NEXUS_SUPABASE_LIVE=1 to load additional documents without redeploying."
        ),
    }
