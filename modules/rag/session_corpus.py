"""Ephemeral per-session document overlays for upload-driven RAG analysis.

Uploaded documents stay in process memory only. The bounded, least-recently-used store is a
session convenience, not durable storage: entries disappear on restart or eviction.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import re
import secrets
import threading

MAX_SESSION_BYTES = 20 * 1024 * 1024
MAX_TOTAL_SESSION_BYTES = 40 * 1024 * 1024
MAX_DOCUMENTS_PER_SESSION = 30
MAX_SESSIONS = 128


class SessionCorpus:
    """Thread-safe, bounded in-memory documents keyed by an opaque browser session id."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._sessions: OrderedDict[str, OrderedDict[str, str]] = OrderedDict()

    def append(self, session_id: str | None, filename: str, text: str) -> tuple[str, str]:
        """Append a document and return ``(session_id, stored_source_name)``."""
        sid = (session_id or "").strip() or secrets.token_urlsafe(24)
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", sid):
            raise ValueError("session_id must contain 1–128 letters, numbers, hyphens, or underscores.")
        clean_name = " ".join(Path(filename.replace("\\", "/")).name.split())[:200]
        if not clean_name or clean_name in {".", ".."}:
            clean_name = "uploaded-document.txt"
        if not text.strip():
            raise ValueError("The uploaded document contains no readable text.")
        incoming_bytes = len(text.encode("utf-8"))
        if incoming_bytes > MAX_SESSION_BYTES:
            raise ValueError(
                f"This document exceeds the {MAX_SESSION_BYTES // (1024 * 1024)} MB per-session limit."
            )

        with self._lock:
            documents = self._sessions.setdefault(sid, OrderedDict())
            current_bytes = sum(len(value.encode("utf-8")) for value in documents.values())
            if current_bytes + incoming_bytes > MAX_SESSION_BYTES:
                raise ValueError(
                    f"This session's in-memory corpus is capped at {MAX_SESSION_BYTES // (1024 * 1024)} MB. "
                    "Start a new session to upload a fresh set of documents."
                )
            if len(documents) >= MAX_DOCUMENTS_PER_SESSION:
                raise ValueError(
                    f"A session can contain at most {MAX_DOCUMENTS_PER_SESSION} uploaded documents. "
                    "Start a new session to upload a fresh set."
                )

            def total_bytes() -> int:
                return sum(
                    len(value.encode("utf-8"))
                    for session_documents in self._sessions.values()
                    for value in session_documents.values()
                )

            # Evict the least-recently-used *other* sessions to preserve the bounded process-memory
            # limit. Never silently discard documents from the session currently being extended.
            while len(self._sessions) > MAX_SESSIONS or total_bytes() + incoming_bytes > MAX_TOTAL_SESSION_BYTES:
                victim = next((key for key in self._sessions if key != sid), None)
                if victim is None:
                    raise ValueError(
                        f"The in-memory RAG session store is capped at "
                        f"{MAX_TOTAL_SESSION_BYTES // (1024 * 1024)} MB. "
                        "Try again after the service restarts."
                    )
                del self._sessions[victim]

            source = clean_name
            path = Path(clean_name)
            suffix = path.suffix
            stem = path.stem
            serial = 2
            while source in documents:
                source = f"{stem}-{serial}{suffix}"
                serial += 1
            documents[source] = text
            self._sessions.move_to_end(sid)
            return sid, source

    def documents(self, session_id: str | None) -> list[tuple[str, str]]:
        """Return a snapshot in upload order; callers cannot mutate the store."""
        if not session_id:
            return []
        with self._lock:
            documents = self._sessions.get(session_id)
            if documents is None:
                return []
            self._sessions.move_to_end(session_id)
            return list(documents.items())

    def clear(self) -> None:
        """Clear all sessions. Intended for isolated tests and process shutdown."""
        with self._lock:
            self._sessions.clear()


SESSION_CORPUS = SessionCorpus()
