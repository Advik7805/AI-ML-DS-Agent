"""Experiment tracking must record runs without ever being able to break training."""

from __future__ import annotations

import json

import pytest


def test_tracking_degrades_to_jsonl_without_mlflow(monkeypatch, tmp_path) -> None:
    """No MLFLOW_TRACKING_URI means a local log, and status() says so rather than hiding it."""
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setenv("SAROS_RUNS_FILE", str(tmp_path / "runs.jsonl"))

    import importlib

    from core import tracking

    importlib.reload(tracking)
    st = tracking.status()
    assert st["backend"] == "jsonl"
    assert st["mlflow_uri"] is None
    assert st["warning"] and "MLFLOW_TRACKING_URI" in st["warning"]


def test_log_run_writes_a_queryable_record(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setenv("SAROS_RUNS_FILE", str(tmp_path / "runs.jsonl"))

    import importlib

    from core import tracking

    importlib.reload(tracking)

    rec = tracking.log_run("churn", {"pr_auc": 0.6771, "n_samples": 834}, {"threshold": 0.41})
    assert rec["body"] == "churn"
    assert rec["metrics"]["pr_auc"] == pytest.approx(0.6771)

    runs = tracking.recent_runs()
    assert len(runs) == 1
    assert runs[0]["run_id"] == rec["run_id"]
    assert runs[0]["params"]["threshold"] == pytest.approx(0.41)


def test_numpy_scalars_survive_serialisation(monkeypatch, tmp_path) -> None:
    """sklearn metrics are numpy floats; json.dumps raises on them. This is a real failure mode."""
    import numpy as np

    monkeypatch.setenv("SAROS_RUNS_FILE", str(tmp_path / "runs.jsonl"))

    import importlib

    from core import tracking

    importlib.reload(tracking)

    rec = tracking.log_run(
        "churn",
        {"pr_auc": np.float64(0.6771), "n": np.int64(834), "arr": np.array([1.0, 2.0])},
    )
    assert rec["metrics"]["pr_auc"] == pytest.approx(0.6771)
    assert rec["metrics"]["n"] == 834
    assert rec["metrics"]["arr"] == [1.0, 2.0]
    # And the file itself must be valid JSONL.
    lines = (tmp_path / "runs.jsonl").read_text().strip().splitlines()
    json.loads(lines[-1])


def test_an_unserialisable_metric_does_not_raise(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("SAROS_RUNS_FILE", str(tmp_path / "runs.jsonl"))

    import importlib

    from core import tracking

    importlib.reload(tracking)

    class Weird:
        pass

    rec = tracking.log_run("churn", {"obj": Weird()})
    assert "obj" in rec["metrics"], "should be coerced to str, not dropped"


def test_an_unwritable_log_does_not_raise(monkeypatch) -> None:
    """A read-only or missing path must not take a training run down with it."""
    monkeypatch.setenv("SAROS_RUNS_FILE", "/proc/definitely/not/writable/runs.jsonl")

    import importlib

    from core import tracking

    importlib.reload(tracking)

    rec = tracking.log_run("churn", {"pr_auc": 0.5})
    assert rec["body"] == "churn"
    assert "jsonl_error" in rec, "the failure must be reported, not swallowed silently"


def test_every_body_train_is_tracked(registry, monkeypatch, tmp_path) -> None:
    """The __init_subclass__ hook is what makes this automatic. Prove it covers every body.

    A body whose train() is not wrapped would simply never appear in the log — the failure is
    invisible unless something asserts on it.
    """
    monkeypatch.setenv("SAROS_RUNS_FILE", str(tmp_path / "runs.jsonl"))

    import importlib

    from core import tracking

    importlib.reload(tracking)

    for bid in ("churn", "fraud", "maintenance"):
        registry.get(bid).train()

    bodies = {r["body"] for r in tracking.recent_runs(limit=50)}
    assert bodies == {"churn", "fraud", "maintenance"}
    for r in tracking.recent_runs(limit=50):
        assert r["metrics"]["train_seconds"] >= 0, "duration must be recorded"
        assert r["metrics"]["n_samples"] > 0


def test_tracking_cannot_break_training(monkeypatch, tmp_path) -> None:
    """If log_run raises for any reason, train() must still return a report."""
    monkeypatch.setenv("SAROS_RUNS_FILE", str(tmp_path / "runs.jsonl"))

    from core import tracking

    def boom(*a, **k):
        raise RuntimeError("tracker exploded")

    monkeypatch.setattr(tracking, "log_run", boom)

    report = registry_module_train("churn")
    assert report.n_samples > 0, "training must survive a broken tracker"


def registry_module_train(body: str):
    from core.contract import ModuleRegistry

    return ModuleRegistry().discover("modules").get(body).train()


# --- orchestration ------------------------------------------------------------


def test_trainable_bodies_is_derived_not_hardcoded() -> None:
    """It was a literal tuple for one commit and was already wrong (pricing raises)."""
    from core.contract import Module, ModuleRegistry

    from flows.train_all import TRAINABLE

    registry = ModuleRegistry().discover("modules")
    for bid in TRAINABLE:
        own = type(registry.get(bid)).__dict__.get("train")
        assert getattr(own, "__wrapped__", own) is not Module.train, f"{bid} does not implement train()"

    # rag and scraper are stateless and must not be in the list.
    assert "rag" not in TRAINABLE and "scraper" not in TRAINABLE
    assert "churn" in TRAINABLE


def test_the_orchestrator_works_without_prefect() -> None:
    """The whole point: Prefect is optional, so this must run in any environment."""
    from flows.train_all import HAVE_PREFECT, main

    # Whether or not Prefect is installed, main() must accept and run the list.
    rc = main(["churn"])
    assert rc == 0
    assert isinstance(HAVE_PREFECT, bool)


def test_the_orchestrator_rejects_unknown_bodies() -> None:
    from flows.train_all import main

    assert main(["definitely_not_a_body"]) == 2


def test_a_failing_body_does_not_stop_the_others() -> None:
    """train_one must catch, so one broken body cannot abort the run."""
    from flows.train_all import train_one

    result = train_one("rag")  # stateless: no train()
    assert result["ok"] is False
    assert "error" in result
