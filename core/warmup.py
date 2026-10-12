"""Warm-up: train every body in the background while `/health` stays instant.

The problem this solves is specific and visible in a real trace: bodies train lazily on first
use, so the *first* request after a cold start pays for the training. On Render's free tier that
is a cold start plus a 14+ second tool call, which is where a user decides the product is broken.

So the lifespan hook starts one daemon thread that walks every trainable body and touches its
cached state. `/health` never waits for it — a health check that blocks on model training is a
health check that fails the deploy.

Two rules:

* **Idempotent.** Warming goes through the same ``bundle`` property a request would use, so the
  module's own lock serialises it and a user request that arrives mid-warm either waits for the
  artifact or trains it itself. Warming never trains a *second* copy.
* **Visible.** The per-body outcome (ready / failed, seconds, headline score) is published on
  ``GET /api/v1/warmup``. A silent warm-up leaves you guessing whether the 14 seconds you are
  still seeing is a cold model or something else.
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from core.contract import ModuleRegistry

_LOCK = threading.Lock()
_STATE: dict[str, Any] = {
    "status": "idle",  # idle | warming | done
    "started_at": None,
    "finished_at": None,
    "seconds": None,
    "bodies": {},
}


def _set(**kwargs: Any) -> None:
    with _LOCK:
        _STATE.update(kwargs)


def _set_body(body: str, payload: dict[str, Any]) -> None:
    with _LOCK:
        _STATE["bodies"][body] = payload


def state() -> dict[str, Any]:
    """A copy of the warm-up state, safe to serialise into a response."""
    with _LOCK:
        return {
            "status": _STATE["status"],
            "started_at": _STATE["started_at"],
            "finished_at": _STATE["finished_at"],
            "seconds": _STATE["seconds"],
            "bodies": {k: dict(v) for k, v in _STATE["bodies"].items()},
        }


def reset() -> None:
    """Test helper: back to a pristine idle state."""
    with _LOCK:
        _STATE.update(
            {"status": "idle", "started_at": None, "finished_at": None, "seconds": None, "bodies": {}}
        )


def trainable_bodies() -> tuple[str, ...]:
    """The training set, from the orchestrator rather than a second hardcoded list."""
    from flows.train_all import TRAINABLE

    return TRAINABLE


def _headline(module: Any) -> str:
    """Whatever number this body is judged on, so the endpoint reports something meaningful."""
    bundle = getattr(module, "_bundle", None)
    if bundle is None:
        return ""
    for attr in ("metrics",):
        metrics = getattr(bundle, attr, None)
        if isinstance(metrics, dict) and metrics:
            for key in ("pr_auc", "r2", "roc_auc"):
                if key in metrics:
                    return f"{key}={metrics[key]:.4f}"
    for key in ("n_rows", "n_train"):
        value = getattr(bundle, key, None)
        if value:
            return f"rows={value}"
    return ""


def warm_one(module: Any, *, deep: bool = False) -> dict[str, Any]:
    """Make one body ready using the same lazily-built state a request would touch.

    ``deep`` also materialises the evaluation, because the model card is the first thing a body
    page and the hub run log read. Bundles carry their report, so this is normally a no-op.
    """
    started = time.perf_counter()
    try:
        # A body with a trained bundle trains or loads it here. The properties are internally
        # locked, so this is safe to race with a live request.
        for attr in ("bundle", "model", "retriever"):
            if hasattr(type(module), attr):
                getattr(module, attr)
                break
        if deep:
            try:
                module.evaluate()
            except NotImplementedError:
                pass  # a body with nothing to evaluate (scraper) is not a failed warm-up
        return {
            "ok": True,
            "seconds": round(time.perf_counter() - started, 2),
            "headline": _headline(module),
        }
    except Exception as exc:  # noqa: BLE001 - warm-up must never take the app down
        return {
            "ok": False,
            "seconds": round(time.perf_counter() - started, 2),
            "error": f"{type(exc).__name__}: {exc}",
        }


def warm(registry: "ModuleRegistry", bodies: tuple[str, ...] | None = None) -> dict[str, Any]:
    """Train/warm every body in sequence. Synchronous; call it from a thread.

    Sequenced rather than parallel on purpose: seven concurrent trainings on a 512 MB / 1 CPU
    instance is how you turn a slow first request into an out-of-memory kill.
    """
    targets = bodies if bodies is not None else trainable_bodies()
    # Stateless bodies (rag's index, pricing's fit) are cheap and worth warming too, so the first
    # knowledge question and the first price quote are both instant.
    extra = tuple(b for b in registry.ids() if b not in targets)

    _set(status="warming", started_at=time.time(), bodies={})
    for body in (*targets, *extra):
        try:
            module = registry.get(body)
        except KeyError:
            _set_body(body, {"ok": False, "error": "not in registry"})
            continue
        result = warm_one(module, deep=body in targets)
        result["role"] = "trained" if body in targets else "lazy assets"
        _set_body(body, result)

    finished = time.time()
    with _LOCK:
        started = _STATE["started_at"] or finished
        _STATE.update(
            {
                "status": "done",
                "finished_at": finished,
                "seconds": round(finished - started, 2),
            }
        )
    return state()


def start_background(registry: "ModuleRegistry") -> threading.Thread:
    """Kick off warm-up on a daemon thread and return it (for tests and for the lifespan)."""
    thread = threading.Thread(
        target=warm, args=(registry,), name="saros-warmup", daemon=True
    )
    _set(status="warming", started_at=time.time(), bodies={})
    thread.start()
    return thread
