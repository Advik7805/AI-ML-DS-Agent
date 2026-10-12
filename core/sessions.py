"""Session-scoped state: what *you* uploaded, held only in memory.

**The rule this file exists to enforce:** an upload is never written to disk, never persisted,
and never merged into the shipped corpus. It lives in this process, for this session, and it is
gone when the process recycles. On Render's free tier that means a redeploy clears it, which is
the correct behaviour for a demo that people drop their own files into — a stranger's invoice
should not survive into the next visitor's knowledge base.

What it buys, though, is real: the body you uploaded to scores your file with its actual model,
the hub agent receives the file's profile as context, and a document you upload to the Knowledge
body is retrieved *per question* with a citation back to your filename.

Sized deliberately: 10 MB per file (enforced at the API), 12 uploads and 24 MB per session,
6-hour idle TTL, 40 sessions, LRU-evicted. A free tier has 512 MB and one worker, so an
unbounded dict here is a scheduled outage.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

#: Enforced at the API boundary too; duplicated here so the store cannot be bypassed.
MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_UPLOADS_PER_SESSION = 12
MAX_SESSION_BYTES = 24 * 1024 * 1024
SESSION_TTL_SECONDS = 6 * 60 * 60
MAX_SESSIONS = 40


@dataclass
class UploadRecord:
    """One file a user dropped in, plus what the body made of it."""

    upload_id: str
    filename: str
    module_id: str
    size_bytes: int
    kind: str = "unknown"
    rows: int | None = None
    columns: list[str] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def as_dict(self) -> dict[str, Any]:
        return {
            "upload_id": self.upload_id,
            "filename": self.filename,
            "module_id": self.module_id,
            "size_bytes": self.size_bytes,
            "kind": self.kind,
            "rows": self.rows,
            "columns": self.columns,
            "summary": self.summary,
            "created_at": self.created_at,
        }


@dataclass
class Session:
    """Everything one browser session has handed the platform."""

    id: str
    created_at: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    uploads: list[UploadRecord] = field(default_factory=list)
    #: (source, heading, text) triples from uploaded documents. Retrieval-in, never persisted.
    rag_passages: list[tuple[str, str, str]] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def bytes_used(self) -> int:
        return sum(u.size_bytes for u in self.uploads)

    def add(self, record: UploadRecord) -> None:
        with self.lock:
            self.uploads.append(record)
            # Oldest first out: a session cannot grow into a memory problem.
            while len(self.uploads) > MAX_UPLOADS_PER_SESSION or self.bytes_used > MAX_SESSION_BYTES:
                self.uploads.pop(0)

    def add_rag_passages(self, source: str, chunks: list[tuple[str, str]]) -> int:
        """Append an uploaded document's chunks under a filename-scoped source label."""
        with self.lock:
            for heading, text in chunks:
                self.rag_passages.append((source, heading, text))
            # Keep the overlay bounded: this is a demo overlay, not a document store.
            if len(self.rag_passages) > 400:
                self.rag_passages = self.rag_passages[-400:]
            return len(self.rag_passages)

    def profiles(self) -> list[dict[str, Any]]:
        return [
            {
                "filename": u.filename,
                "body": u.module_id,
                "kind": u.kind,
                "rows": u.rows,
                "columns": u.columns[:40],
                "summary": u.summary,
            }
            for u in self.uploads
        ]

    def context_block(self) -> str:
        """The text the hub agent receives so it can actually talk about your file.

        Deliberately factual and small: row count, columns, the summary the body computed, and
        the numbers it produced. No file contents are pasted in wholesale — a 10 MB CSV would
        blow any free-tier context window, and a partial paste would let the model quote rows
        that are not in the data.
        """
        if not self.uploads:
            return ""
        lines = [
            "SESSION CONTEXT — files this user uploaded in this browser session. These are NOT "
            "part of the shipped tenant data; they are the user's own files, processed in memory "
            "and never persisted. When you answer about them, name the file.",
        ]
        for u in self.uploads[-4:]:
            cols = ", ".join(u.columns[:24]) or "(no tabular columns)"
            lines.append(
                f"- `{u.filename}` uploaded to the **{u.module_id}** body: {u.rows if u.rows is not None else 'n/a'} rows, "
                f"columns: {cols}."
            )
            if u.summary:
                for k, v in list(u.summary.items())[:8]:
                    lines.append(f"    · {k}: {v}")
        if self.rag_passages:
            lines.append(
                "- Uploaded documents are also searchable in the Knowledge body for this "
                "session; relevant passages from them may appear in your context below."
            )
        return "\n".join(lines)


class SessionStore:
    """In-memory, TTL'd, LRU-capped. One instance per process."""

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    @staticmethod
    def new_id() -> str:
        return f"s_{uuid.uuid4().hex[:20]}"

    @staticmethod
    def valid_id(candidate: str | None) -> bool:
        if not candidate:
            return False
        if not (8 <= len(candidate) <= 64):
            return False
        return all(c.isalnum() or c in "-_" for c in candidate)

    def _sweep(self, now: float) -> None:
        dead = [sid for sid, s in self._sessions.items() if now - s.last_seen > SESSION_TTL_SECONDS]
        for sid in dead:
            self._sessions.pop(sid, None)
        if len(self._sessions) > MAX_SESSIONS:
            oldest = sorted(self._sessions.values(), key=lambda s: s.last_seen)[
                : len(self._sessions) - MAX_SESSIONS
            ]
            for s in oldest:
                self._sessions.pop(s.id, None)

    def get(self, session_id: str | None) -> Session | None:
        if not self.valid_id(session_id):
            return None
        assert session_id is not None
        with self._lock:
            now = time.time()
            self._sweep(now)
            session = self._sessions.get(session_id)
            if session is not None:
                session.last_seen = now
            return session

    def get_or_create(self, session_id: str | None) -> Session:
        """Reuse a valid id, else mint one. An invalid or expired id is replaced, not rejected:
        the user gets a working session and the response says which id is now in force."""
        existing = self.get(session_id)
        if existing is not None:
            return existing
        sid = session_id if self.valid_id(session_id) else self.new_id()
        assert sid is not None
        with self._lock:
            now = time.time()
            self._sweep(now)
            session = Session(id=sid)
            self._sessions[sid] = session
            return session

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "n_sessions": len(self._sessions),
                "ttl_seconds": SESSION_TTL_SECONDS,
                "max_file_bytes": MAX_FILE_BYTES,
                "storage": "in-memory only; nothing is written to disk",
            }


#: Process-wide store. Imported by the API, which is the only writer.
STORE = SessionStore()
