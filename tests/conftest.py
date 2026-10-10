"""Shared fixtures.

The registry fixture lives here rather than in a single test module so every suite sees the
same discovered platform. It is module-scoped: discovery imports all seven bodies and trains
several of them on first use, which is far too slow to repeat per test.
"""

from __future__ import annotations

import pytest

from core.contract import ModuleRegistry


@pytest.fixture(scope="module")
def registry() -> ModuleRegistry:
    """Every body, discovered. A load error here is a test failure, not a skipped body."""
    reg = ModuleRegistry().discover("modules")
    assert getattr(reg, "load_errors", {}) == {}
    return reg
