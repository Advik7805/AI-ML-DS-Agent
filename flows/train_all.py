"""Training orchestration: Prefect when installed, plain functions when not.

Seven bodies, each trained independently, with no record of which succeeded — that is what this
replaces. One command trains everything that can be trained and reports what happened.

The design constraint is that Prefect must be optional. It is a real dependency with a real
weight, and the API image has neither the room nor the need for it. So the orchestration is
plain Python first: `train_all()` is an ordinary function that works in any environment, and
when Prefect is importable the same work is exposed as `@flow`/`@task` so you get Prefect's UI,
retries and history. Nothing here degrades if Prefect is missing; the decorators simply are not
applied.

Usage:

    python -m flows.train_all              # train every body, print a table
    python -m flows.train_all churn fraud  # train only these

With Prefect installed, the flow registers under the name "saros-train-all" and can be run from
a Prefect server or on a schedule.
"""

from __future__ import annotations

import sys
import time
from typing import Any, Callable

def _trainable_bodies() -> tuple[str, ...]:
    """Bodies that actually implement train(), discovered rather than listed.

    This was a hardcoded tuple for one commit and was already wrong: pricing was in it and
    raises NotImplementedError. A list like that only rots — rag and scraper are stateless and
    pricing fits its demand model lazily, so none of them belong here. Deriving it means a body
    that later gains a train() is picked up automatically.
    """
    from core.contract import Module, ModuleRegistry

    registry = ModuleRegistry().discover("modules")
    out = []
    for bid in sorted(registry.ids()):
        own = type(registry.get(bid)).__dict__.get("train")
        # __init_subclass__ wraps train() for tracking, so look through the wrapper.
        inner = getattr(own, "__wrapped__", None)
        if inner is not None and inner is not Module.train:
            out.append(bid)
    return tuple(out)


TRAINABLE: tuple[str, ...] = _trainable_bodies()

try:
    from prefect import flow, task

    HAVE_PREFECT = True
except ImportError:  # pragma: no cover - the common case on the API image

    def flow(*dargs: Any, **dkwargs: Any) -> Callable[[Callable], Callable]:
        """No-op stand-in so the module works identically without Prefect."""

        def deco(fn: Callable) -> Callable:
            return fn

        return deco if not dargs or callable(dargs[0]) is False else (dargs[0] if callable(dargs[0]) else deco)

    def task(*dargs: Any, **dkwargs: Any) -> Callable[[Callable], Callable]:
        def deco(fn: Callable) -> Callable:
            return fn

        return deco if not dargs or callable(dargs[0]) is False else (dargs[0] if callable(dargs[0]) else deco)

    HAVE_PREFECT = False


@task(name="train-body", retries=1, retry_delay_seconds=5)
def train_one(body: str) -> dict[str, Any]:
    """Train one body and report the outcome. Never raises — a bad body must not stop the rest."""
    from core.contract import ModuleRegistry

    t0 = time.perf_counter()
    try:
        report = ModuleRegistry().discover("modules").get(body).train()
    except Exception as exc:  # noqa: BLE001
        return {
            "body": body, "ok": False, "seconds": round(time.perf_counter() - t0, 2),
            "error": f"{type(exc).__name__}: {exc}",
        }

    # Different bodies report different headline metrics; take whichever this one has.
    headline = next(
        (f"{k}={report.scores[k]:.4f}" for k in ("pr_auc", "r2", "roc_auc") if k in report.scores),
        "",
    )
    return {
        "body": body, "ok": True, "seconds": round(time.perf_counter() - t0, 2),
        "headline": headline, "n_samples": report.n_samples,
        "threshold": report.threshold, "source": getattr(report, "group_basis", "") or "",
    }


@flow(name="saros-train-all", log_prints=True)
def train_all(bodies: tuple[str, ...] = TRAINABLE) -> list[dict[str, Any]]:
    """Train the given bodies in sequence and return one record per body."""
    results = [train_one(b) for b in bodies]

    ok = sum(1 for r in results if r["ok"])
    print(f"\n{ok}/{len(results)} bodies trained in {sum(r['seconds'] for r in results):.1f}s")
    for r in results:
        if r["ok"]:
            print(f"  ok    {r['body']:12s} {r['seconds']:6.2f}s  {r.get('headline','')}")
        else:
            print(f"  FAIL  {r['body']:12s} {r['error']}")
    return results


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    bodies = tuple(args) if args else TRAINABLE

    unknown = [b for b in bodies if b not in TRAINABLE]
    if unknown:
        print(f"not trainable: {', '.join(unknown)}")
        print(f"trainable:     {', '.join(TRAINABLE)}")
        return 2

    print(f"orchestrator: {'prefect' if HAVE_PREFECT else 'plain (prefect not installed)'}")
    print(f"training:     {', '.join(bodies)}\n")
    results = train_all(bodies)
    return 0 if all(r["ok"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
