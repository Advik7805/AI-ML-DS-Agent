"""Contract conformance — the rules that keep the bodies genuinely independent.

These tests are the enforcement half of ``docs/ARCHITECTURE.md`` §1. Without them the
"independent bodies" claim is just a folder layout.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

from core.contract import Module, ModuleRegistry
from core.types import AgentTool, ModuleManifest, Prediction

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULES_DIR = REPO_ROOT / "modules"


def module_packages() -> list[str]:
    return sorted(
        d.name for d in MODULES_DIR.iterdir()
        if d.is_dir() and not d.name.startswith("_") and (d / "__init__.py").exists()
    )


@pytest.fixture(scope="module")
def registry() -> ModuleRegistry:
    return ModuleRegistry().discover("modules")


# --- discovery ---------------------------------------------------------------


def test_discovery_finds_every_package(registry: ModuleRegistry) -> None:
    """Every package under modules/ must be registered, or reported as a load error.

    The registry keys on ``MODULE.id`` while load errors key on package name, so the two are
    compared separately rather than unioned — a module whose id differs from its directory
    (hyphens are not legal in Python identifiers) is legitimate and must not look like a miss.
    """
    failed = set(getattr(registry, "load_errors", {}))
    registered_ids = set(registry.ids())
    unaccounted = []
    for pkg in module_packages():
        if pkg in failed:
            continue
        mod = importlib.import_module(f"modules.{pkg}")
        instance = getattr(mod, "MODULE", None)
        if instance is None or instance.id not in registered_ids:
            unaccounted.append(pkg)
    assert not unaccounted, f"packages neither registered nor error-reported: {unaccounted}"


def test_discovery_has_no_errors(registry: ModuleRegistry) -> None:
    assert getattr(registry, "load_errors", {}) == {}


def test_at_least_one_module_loaded(registry: ModuleRegistry) -> None:
    assert len(registry) >= 1


# --- contract conformance ----------------------------------------------------


def test_every_module_satisfies_the_contract(registry: ModuleRegistry) -> None:
    for module in registry.all():
        assert isinstance(module, Module), f"{module.id} is not a Module subclass"
        assert module.id, "module id must be non-empty"
        manifest = module.manifest
        assert isinstance(manifest, ModuleManifest)
        assert manifest.id == module.id, f"manifest.id != class id for {module.id}"
        assert manifest.title and manifest.summary
        assert manifest.routes.get("ui", "").startswith("/"), f"{module.id} needs a ui route"
        assert manifest.routes.get("api", "").startswith("/api/"), f"{module.id} needs an api route"
        # Accent must be a hex colour — the hub uses it to theme each body.
        assert manifest.accent.startswith("#") and len(manifest.accent) == 7


def test_ui_routes_are_unique(registry: ModuleRegistry) -> None:
    """Two bodies claiming the same route would silently shadow each other in the hub."""
    routes = [m.manifest.routes["ui"] for m in registry.all()]
    assert len(routes) == len(set(routes)), f"duplicate ui routes: {routes}"


def test_tool_names_are_globally_unique(registry: ModuleRegistry) -> None:
    """The agent dispatches by bare tool name, so a collision would route to the wrong body."""
    names = [t.name for t in registry.all_tools()]
    assert len(names) == len(set(names)), f"duplicate tool names: {names}"


def test_tools_are_valid_json_schema(registry: ModuleRegistry) -> None:
    for tool in registry.all_tools():
        assert isinstance(tool, AgentTool)
        assert tool.description, f"tool {tool.name} needs a description for the LLM"
        assert tool.parameters_schema.get("type") == "object"
        assert isinstance(tool.parameters_schema.get("properties"), dict)


def test_every_declared_tool_is_dispatchable(registry: ModuleRegistry) -> None:
    """A tool in the schema with no ``tool_<name>`` method is a runtime 500 waiting to happen."""
    for module in registry.all():
        for tool in module.tools():
            assert hasattr(module, f"tool_{tool.name}"), (
                f"{module.id} declares tool '{tool.name}' but has no tool_{tool.name} method"
            )


def test_call_tool_rejects_unknown_names(registry: ModuleRegistry) -> None:
    module = registry.all()[0]
    with pytest.raises(KeyError):
        module.call_tool("definitely_not_a_real_tool", {})


def test_registry_rejects_duplicate_ids() -> None:
    class Dummy(Module):
        id = "dupe"

        @property
        def manifest(self) -> ModuleManifest:
            return ModuleManifest(id="dupe", title="d", summary="s", routes={"ui": "/d", "api": "/api/d"})

        def predict(self, payload):  # pragma: no cover
            return Prediction(module_id="dupe", score=0.0, label="l", decision="d")

        def tools(self):  # pragma: no cover
            return []

    reg = ModuleRegistry()
    reg.register(Dummy())
    with pytest.raises(ValueError, match="Duplicate module id"):
        reg.register(Dummy())


def test_registry_rejects_id_mismatch() -> None:
    class Bad(Module):
        id = "one"

        @property
        def manifest(self) -> ModuleManifest:
            return ModuleManifest(id="two", title="t", summary="s")

        def predict(self, payload):  # pragma: no cover
            raise NotImplementedError

        def tools(self):  # pragma: no cover
            return []

    with pytest.raises(ValueError, match="mismatch"):
        ModuleRegistry().register(Bad())


# --- independence ------------------------------------------------------------


@pytest.mark.parametrize("pkg", module_packages())
def test_module_does_not_import_another_module(pkg: str) -> None:
    """The one rule that makes the bodies independent rather than nominally so.

    A module may import ``core`` and the stdlib, but never ``modules.<other>``. Cross-module
    imports are how a modular monolith quietly becomes a ball of mud.
    """
    pkg_dir = MODULES_DIR / pkg
    offenders: list[str] = []
    for py in pkg_dir.rglob("*.py"):
        tree = ast.parse(py.read_text(), filename=str(py))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module] if node.module else []
            else:
                continue
            for name in names:
                if not name:
                    continue
                parts = name.split(".")
                if parts[0] == "modules" and len(parts) > 1 and parts[1] != pkg:
                    offenders.append(f"{py.relative_to(REPO_ROOT)} imports {name}")
    assert not offenders, "\n".join(offenders)


@pytest.mark.parametrize("pkg", module_packages())
def test_module_exports_a_MODULE_singleton(pkg: str) -> None:
    init = (MODULES_DIR / pkg / "__init__.py").read_text()
    assert "MODULE" in init, f"modules/{pkg}/__init__.py must export MODULE"


def test_discovery_is_idempotent() -> None:
    """uvicorn --reload and a second test module both re-run discovery.

    A second pass must not report the same singleton as a duplicate — that was turning
    /health into a wall of spurious load errors.
    """
    reg = ModuleRegistry().discover("modules")
    first = reg.ids()
    errors = getattr(reg, "load_errors", {})
    reg.discover("modules")
    assert reg.ids() == first
    assert getattr(reg, "load_errors", {}) == errors
    assert errors == {}
