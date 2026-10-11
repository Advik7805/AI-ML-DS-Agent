"""Experiment tracking: MLflow when configured, a local JSONL log when not.

The honest version of this feature. MLflow is worth having — seven bodies, each selecting an
estimator by a metric, is exactly the situation where "which run produced the model I shipped"
becomes unanswerable. But MLflow and its dependency tree are ~100 MB, and the API runs on a
512 MB free container where that is not a rounding error.

So: tracking is always on, and it degrades. With `MLFLOW_TRACKING_URI` set and mlflow
importable, runs go to MLflow. Otherwise every run is appended to `artifacts/runs.jsonl`, which
is queryable with one line of pandas and survives a redeploy if you mount a volume. Either way
the training path records the same facts, and no caller has to know which backend answered.

mlflow is imported lazily, inside the function that needs it. It is deliberately not in
requirements.txt — see requirements-mlflow.txt.
"""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

RUNS_FILE = Path(os.getenv("SAROS_RUNS_FILE", "artifacts/runs.jsonl"))


def _mlflow():
    """Return the mlflow module if it is installed and configured, else None."""
    uri = os.getenv("MLFLOW_TRACKING_URI", "").strip()
    if not uri:
        return None
    try:
        import mlflow
    except ImportError:
        return None
    try:
        mlflow.set_tracking_uri(uri)
    except Exception:
        return None
    return mlflow


def backend() -> str:
    """Which backend `log_run` will use. Surfaced by the API so this is never a guess."""
    if _mlflow() is not None:
        return "mlflow"
    return "jsonl"


def status() -> dict[str, Any]:
    ml = _mlflow()
    n_local = 0
    if RUNS_FILE.exists():
        try:
            n_local = sum(1 for _ in RUNS_FILE.open())
        except OSError:
            pass
    return {
        "backend": backend(),
        "mlflow_uri": os.getenv("MLFLOW_TRACKING_URI", "") or None,
        "runs_file": str(RUNS_FILE),
        "runs_file_entries": n_local,
        "warning": None
        if ml is not None
        else (
            "Tracking to a local JSONL file. Set MLFLOW_TRACKING_URI and install mlflow "
            "(requirements-mlflow.txt) to send runs to an MLflow server."
        ),
    }


def log_run(
    body: str,
    metrics: dict[str, Any],
    params: dict[str, Any] | None = None,
    tags: dict[str, str] | None = None,
    *,
    experiment: str | None = None,
) -> dict[str, Any]:
    """Record one training run. Never raises — tracking must not be able to break training.

    Returns the record that was written, so callers can surface the run id.
    """
    record = {
        "run_id": f"{body}-{int(time.time() * 1000)}",
        "body": body,
        "experiment": experiment or f"saros-{body}",
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "backend": backend(),
        "metrics": _clean(metrics),
        "params": _clean(params or {}),
        "tags": {"body": body, **(tags or {})},
    }

    ml = _mlflow()
    if ml is not None:
        try:
            ml.set_experiment(record["experiment"])
            with ml.start_run(run_name=record["run_id"]) as run:
                ml.log_metrics({k: float(v) for k, v in record["metrics"].items()
                                if isinstance(v, (int, float))})
                ml.log_params({k: str(v) for k, v in record["params"].items()})
                ml.set_tags(record["tags"])
                record["mlflow_run_id"] = run.info.run_id
        except Exception as exc:  # noqa: BLE001
            # An unreachable MLflow server must not fail a training run.
            record["mlflow_error"] = f"{type(exc).__name__}: {exc}"

    try:
        RUNS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with RUNS_FILE.open("a") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
    except OSError as exc:
        record["jsonl_error"] = f"{type(exc).__name__}: {exc}"

    return record


def recent_runs(limit: int = 20, body: str | None = None) -> list[dict[str, Any]]:
    """The local run log, newest first. Works with or without MLflow."""
    if not RUNS_FILE.exists():
        return []
    out: list[dict[str, Any]] = []
    try:
        for line in RUNS_FILE.read_text().splitlines()[-(limit * 4):]:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if body is None or rec.get("body") == body:
                out.append(rec)
    except OSError:
        return []
    return out[-limit:][::-1]


@contextmanager
def timed_run(body: str, **kwargs: Any) -> Iterator[dict[str, Any]]:
    """Context manager that adds wall-clock duration to the recorded metrics."""
    t0 = time.perf_counter()
    holder: dict[str, Any] = {"metrics": {}, "params": {}}
    try:
        yield holder
    finally:
        merged = dict(holder.get("metrics") or {})
        merged["train_seconds"] = round(time.perf_counter() - t0, 3)
        log_run(body, merged, holder.get("params") or {}, **kwargs)


def _clean(d: dict[str, Any]) -> dict[str, Any]:
    """Make a metrics/params dict JSON-serialisable. numpy scalars are the usual problem."""
    import numpy as np

    out = {}
    for k, v in d.items():
        if isinstance(v, (np.integer,)):
            out[k] = int(v)
        elif isinstance(v, (np.floating,)):
            out[k] = float(v)
        elif isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, dict):
            out[k] = _clean(v)
        else:
            try:
                json.dumps(v)
                out[k] = v
            except (TypeError, ValueError):
                out[k] = str(v)
    return out
