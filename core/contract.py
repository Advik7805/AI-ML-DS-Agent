"""The module contract.

The whole platform hinges on this file. Six enterprise modules + an agent + a scraper is only
buildable if every module implements one interface — then the hub can list and launch any
body, the agent can call any body as a tool, and the frontend can render any body's output
with shared components.

Rules enforced by CI (see ``tests/test_contract.py``):

1. A module lives in ``modules/<id>/`` and exports a ``MODULE`` instance.
2. A module may import ``core`` but **never** another module. That single rule is what makes
   the bodies genuinely independent rather than nominally so.
3. ``predict`` returns :class:`~core.types.Prediction`. ``explain`` returns
   :class:`~core.types.Explanation`. No exceptions.
"""

from __future__ import annotations

import functools
import importlib
import pkgutil
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable
from typing import Any

from core.types import AgentTool, Explanation, MetricsReport, ModuleManifest, Prediction, UploadAnalysis


class Module(ABC):
    """Base class every Saros body implements.

    Subclasses must define ``manifest``, ``predict`` and ``tools``. ``explain``, ``evaluate``,
    ``train`` and ``stream`` are optional; the hub degrades gracefully when they are absent.
    """

    #: Filled by the concrete module. Used as the registry key.
    id: str = ""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Wrap any subclass `train` so every training run is recorded automatically.

        Done here rather than in each body for one reason: a body added later cannot forget it.
        The alternative is six call sites and a seventh that silently never gets tracked, which
        is how you end up with a run log that lies by omission.

        Tracking never raises, and `log_run` degrades to a local JSONL file when MLflow is not
        configured, so this cannot turn a working `train()` into a failing one.
        """
        super().__init_subclass__(**kwargs)

        base_train = Module.train
        own = cls.__dict__.get("train")
        if own is None or own is base_train:
            return

        @functools.wraps(own)
        def tracked(self: "Module", config: dict[str, Any] | None = None) -> MetricsReport:
            import time as _time

            t0 = _time.perf_counter()
            report = own(self, config)
            try:
                from core.tracking import log_run

                log_run(
                    self.id,
                    metrics={**report.scores, "n_samples": report.n_samples,
                             "positive_rate": report.positive_rate,
                             "train_seconds": round(_time.perf_counter() - t0, 3)},
                    params={"threshold": report.threshold, "group_basis": report.group_basis,
                            **(config or {})},
                    tags={"source": getattr(self, "source", "") or ""},
                )
            except Exception:  # noqa: BLE001 - tracking must never break training
                pass
            return report

        cls.train = tracked  # type: ignore[method-assign]

    @property
    @abstractmethod
    def manifest(self) -> ModuleManifest:
        """Static description the hub renders as a tile and uses for routing."""

    @abstractmethod
    def predict(self, payload: dict[str, Any]) -> Prediction:
        """Score one record. Must never raise on bad input — return a 422-shaped error instead."""

    @abstractmethod
    def tools(self) -> list[AgentTool]:
        """Capabilities exposed to the hub agent as function-calling tools."""

    # --- optional capabilities -------------------------------------------------

    def explain(self, payload: dict[str, Any]) -> Explanation:
        """Why the model said what it said. Override to add real XAI."""
        return Explanation(kind="none", summary=f"{self.id} does not implement explanations yet.")

    def evaluate(self) -> MetricsReport:
        raise NotImplementedError(f"{self.id} does not implement evaluate()")

    def train(self, config: dict[str, Any] | None = None) -> MetricsReport:
        raise NotImplementedError(f"{self.id} does not implement train()")

    def analyze_upload(
        self,
        filename: str | None,
        content: bytes | None,
        *,
        session_id: str | None = None,
        query: str = "",
    ) -> UploadAnalysis:
        """Analyze an in-memory upload. Bodies opt in and own their format validation."""
        raise NotImplementedError(f"{self.id} does not accept uploaded data")

    async def stream(self) -> AsyncIterator[dict[str, Any]]:
        """Live event feed (fraud transactions, IoT telemetry). Optional."""
        if False:  # pragma: no cover - makes this an async generator
            yield {}
        raise NotImplementedError(f"{self.id} does not implement stream()")

    # --- agent dispatch --------------------------------------------------------

    def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """Dispatch a tool call from the hub agent.

        ``tools()`` only carries JSON Schema; this maps the name onto the real Python method.
        """
        handler: Callable[..., Any] | None = getattr(self, f"tool_{name}", None)
        if handler is None or not callable(handler):
            available = ", ".join(t.name for t in self.tools()) or "none"
            raise KeyError(f"{self.id} has no tool '{name}'. Available: {available}")
        return handler(**arguments)


class ModuleRegistry:
    """Discovers and holds every body. The hub's only view of the platform."""

    def __init__(self) -> None:
        self._modules: dict[str, Module] = {}

    def register(self, module: Module) -> None:
        if not module.id:
            raise ValueError(f"{type(module).__name__} must set a non-empty `id`.")
        if module.id in self._modules:
            raise ValueError(f"Duplicate module id '{module.id}'.")
        if module.manifest.id != module.id:
            raise ValueError(
                f"Module id mismatch: class id='{module.id}' but manifest.id='{module.manifest.id}'."
            )
        self._modules[module.id] = module

    def discover(self, package: str = "modules") -> "ModuleRegistry":
        """Import every ``modules.<name>`` package and register its ``MODULE`` singleton.

        A module that fails to import is skipped with a recorded error rather than taking the
        whole platform down — one broken body must not kill the hub.
        """
        pkg = importlib.import_module(package)
        self.load_errors: dict[str, str] = {}
        for info in pkgutil.iter_modules(pkg.__path__):
            if info.name.startswith("_"):
                continue
            full = f"{package}.{info.name}"
            try:
                mod = importlib.import_module(full)
            except Exception as exc:  # noqa: BLE001 - deliberate: isolate body failures
                self.load_errors[info.name] = f"{type(exc).__name__}: {exc}"
                continue
            instance = getattr(mod, "MODULE", None)
            if instance is None:
                self.load_errors[info.name] = "no MODULE singleton exported"
                continue
            if instance.id in self._modules and self._modules[instance.id] is instance:
                # Idempotent: re-running discovery (uvicorn --reload, a second test module)
                # must not report the same singleton as a duplicate.
                continue
            try:
                self.register(instance)
            except ValueError as exc:
                self.load_errors[info.name] = str(exc)
        return self

    # --- lookups ---------------------------------------------------------------

    def get(self, module_id: str) -> Module:
        try:
            return self._modules[module_id]
        except KeyError:
            raise KeyError(f"Unknown module '{module_id}'. Loaded: {', '.join(sorted(self._modules)) or 'none'}") from None

    def all(self) -> list[Module]:
        return list(self._modules.values())

    def manifests(self) -> list[ModuleManifest]:
        return [m.manifest for m in self._modules.values()]

    def ids(self) -> list[str]:
        return sorted(self._modules)

    def __contains__(self, module_id: object) -> bool:
        return module_id in self._modules

    def __len__(self) -> int:
        return len(self._modules)

    # --- agent surface ---------------------------------------------------------

    def all_tools(self) -> list[AgentTool]:
        """Every tool from every body, flattened, ready to bind onto an LLM."""
        out: list[AgentTool] = []
        for module in self._modules.values():
            out.extend(module.tools())
        return out

    def dispatch(self, tool_name: str, arguments: dict[str, Any] | None = None) -> Any:
        """Route a tool call to whichever body owns it."""
        arguments = arguments or {}
        for module in self._modules.values():
            if any(t.name == tool_name for t in module.tools()):
                return module.call_tool(tool_name, arguments)
        raise KeyError(f"No module owns tool '{tool_name}'. Known: {', '.join(t.name for t in self.all_tools())}")


#: Process-wide registry, populated at app startup.
registry = ModuleRegistry()
