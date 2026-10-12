"""Versioned model artifacts.

Every body that trains caches its bundle to a pickle. That is the right call — retraining on a
cold start is the difference between a demo that answers in a second and one that takes twenty.

It also has a failure mode that is genuinely nasty, and which this module exists to remove:
`pickle` restores whatever fields existed when the object was written, and silently omits any
added since. Add a field to a `@dataclass` bundle — say `reference`, for drift detection — and
every cached artifact on disk loads successfully, looks fine, and then raises `AttributeError`
the first time something touches the new field. The error surfaces at the call site, far from
the cause, long after the change that caused it.

So artifacts carry a schema number. A mismatch raises, and every caller already treats "load
failed" as "retrain" — which is the correct behaviour and requires no special handling.

Bump `SCHEMA` whenever a bundle's fields change. Forgetting to bump it reintroduces exactly the
bug above, so it is asserted in tests.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

#: Bumped whenever any cached bundle's fields change.
#: 3: trainable bundles gained `report`, so a reload no longer has to retrain (or, worse,
#:    invent zeros) to answer "how was this model evaluated?".
SCHEMA = 3


class StaleArtifactError(ValueError):
    """The cached model was written by an older version of the code. Retrain it."""


def save(bundle: Any, path: Path | str) -> Path:
    """Write a bundle with the current schema number stamped on it."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("wb") as fh:
        pickle.dump({"schema": SCHEMA, "bundle": bundle}, fh)
    return p


def load(path: Path | str, expected: int = SCHEMA) -> Any:
    """Read a bundle, refusing anything written under a different schema.

    Raises `StaleArtifactError` on a version mismatch and lets any other load failure propagate
    unchanged — callers already handle a missing or unreadable file by retraining, and this
    should follow the same path rather than needing its own branch everywhere.
    """
    p = Path(path)
    with p.open("rb") as fh:
        blob = pickle.load(fh)  # noqa: S301 - trusted local artifact

    if not isinstance(blob, dict) or "schema" not in blob:
        raise StaleArtifactError(
            f"{p.name} has no schema version (written before versioning was added). Retraining."
        )
    if blob["schema"] != expected:
        raise StaleArtifactError(
            f"{p.name} is schema {blob['schema']}; this code expects {expected}. Retraining."
        )
    return blob["bundle"]
