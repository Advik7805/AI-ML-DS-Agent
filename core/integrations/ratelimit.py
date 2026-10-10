"""Upstash Redis rate limiting — with an honest in-process fallback.

Why this exists: a free-tier backend has a hard request budget, and one crawler or a runaway
agent loop can exhaust it in minutes. The Upstash free tier is 10,000 commands/day, which is
enough to protect the API without costing anything.

The fallback matters as much as the real thing. Without Redis this uses a fixed-window counter
in process memory, which is *weaker* and says so:

- It is per-instance. Two Render instances means twice the effective limit.
- It resets on restart, and a free-tier instance restarts whenever it sleeps.

That is still better than nothing — it stops a single client from monopolising one instance —
but the limitation is reported in the response headers and in `status()` rather than being
papered over. Pretending an in-memory counter is a distributed rate limit is how a service
discovers its real limits during a demo.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

from .config import UPSTASH

DEFAULT_LIMIT = int(os.getenv("NEXUS_RATE_LIMIT", "60") or 60)
DEFAULT_WINDOW = int(os.getenv("NEXUS_RATE_WINDOW", "60") or 60)


@dataclass
class Decision:
    allowed: bool
    remaining: int
    limit: int
    reset_in: int
    backend: str

    def headers(self) -> dict[str, str]:
        return {
            "X-RateLimit-Limit": str(self.limit),
            "X-RateLimit-Remaining": str(max(0, self.remaining)),
            "X-RateLimit-Reset": str(self.reset_in),
            "X-RateLimit-Backend": self.backend,
        }


class InMemoryLimiter:
    """Fixed-window counter, per process. Honest about what it is not."""

    backend = "in-memory"

    def __init__(self, limit: int = DEFAULT_LIMIT, window: int = DEFAULT_WINDOW) -> None:
        self.limit = limit
        self.window = window
        self._counts: dict[str, tuple[int, float]] = {}

    def check(self, key: str) -> Decision:
        now = time.time()
        count, window_start = self._counts.get(key, (0, now))
        if now - window_start >= self.window:
            count, window_start = 0, now
        count += 1
        self._counts[key] = (count, window_start)

        # Opportunistic cleanup: without it a long-lived process accumulates a key per client
        # forever, which on a 512 MB free-tier instance is a slow leak.
        if len(self._counts) > 5000:
            cutoff = now - self.window
            self._counts = {k: v for k, v in self._counts.items() if v[1] > cutoff}

        return Decision(
            allowed=count <= self.limit,
            remaining=self.limit - count,
            limit=self.limit,
            reset_in=max(0, int(self.window - (now - window_start))),
            backend=self.backend,
        )


class UpstashLimiter:
    """Fixed window on Redis via the REST API, so the limit is shared across instances.

    Uses one INCR plus an EXPIRE. That is two commands per request, so at 10,000 commands/day
    the free tier covers ~5,000 requests/day — fine for a portfolio project, and the number is
    the reason `NEXUS_UPSTASH_LIVE` is opt-in rather than automatic.
    """

    backend = "upstash"

    def __init__(self, url: str, token: str, limit: int = DEFAULT_LIMIT,
                 window: int = DEFAULT_WINDOW) -> None:
        self.url = url.rstrip("/")
        self.token = token
        self.limit = limit
        self.window = window

    def check(self, key: str) -> Decision:
        import httpx

        redis_key = f"nexus:rl:{key}"
        headers = {"Authorization": f"Bearer {self.token}"}

        r = httpx.post(
            f"{self.url}/incr/{redis_key}",
            headers=headers,
            # Upstash accepts a command array; EXPIRE keeps abandoned keys from living forever.
            json=[("expire", redis_key, self.window)],
            timeout=5.0,
        )
        r.raise_for_status()
        count = int(r.json().get("result", 0))
        if count == 1:
            httpx.post(
                f"{self.url}/expire/{redis_key}/{self.window}", headers=headers, timeout=5.0
            )

        return Decision(
            allowed=count <= self.limit,
            remaining=self.limit - count,
            limit=self.limit,
            reset_in=self.window,
            backend=self.backend,
        )


_limiter: object | None = None


def get_limiter():
    """Resolve once per process. Never raises — a limiter that fails must not take the API down."""
    global _limiter
    if _limiter is None:
        url = os.getenv("UPSTASH_REDIS_REST_URL", "").strip()
        token = UPSTASH.key
        if UPSTASH.enabled and url and token:
            _limiter = UpstashLimiter(url, token)
        else:
            _limiter = InMemoryLimiter()
    return _limiter


def reset_limiter() -> None:
    """For tests, and for anything that changes the env at runtime."""
    global _limiter
    _limiter = None


def status() -> dict[str, object]:
    limiter = get_limiter()
    live = getattr(limiter, "backend", "in-memory") == "upstash"
    return {
        "backend": getattr(limiter, "backend", "in-memory"),
        "live": live,
        "limit": getattr(limiter, "limit", DEFAULT_LIMIT),
        "window_seconds": getattr(limiter, "window", DEFAULT_WINDOW),
        "scope": "shared across instances" if live else "per process only",
        "warning": None
        if live
        else (
            "Counting in process memory: the limit is per-instance and resets on restart. Set "
            "UPSTASH_REDIS_REST_URL, UPSTASH_REDIS_REST_TOKEN and NEXUS_UPSTASH_LIVE=1 for a "
            "shared limit."
        ),
    }
