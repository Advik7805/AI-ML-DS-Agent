"""Churn & Retention — a Saros body.

Self-contained: own router surface via ``MODULE``, own models, own data, own tests.
May import ``core``; may never import another module (enforced by ``tests/test_contract.py``).
"""

from .service import MODULE, ChurnModule

__all__ = ["MODULE", "ChurnModule"]
