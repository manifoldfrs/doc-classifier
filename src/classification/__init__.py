"""HeronAI – Classification package root.

This *minimal* ``__init__`` re-exports the public pipeline API without
incurring heavy import costs.  Following the repository engineering rules, we
avoid performing substantial work at import-time – the actual orchestrator is
only loaded when the symbols are first accessed.
"""

from __future__ import annotations

from importlib import import_module as _import_module

_pipeline = _import_module(".pipeline", package=__name__)

# Public symbols – classified here for explicitness
classify = _pipeline.classify  # type: ignore[attr-defined]
ClassificationResult = _pipeline.ClassificationResult  # type: ignore[attr-defined]

__all__: list[str] = [
    "classify",
    "ClassificationResult",
]
