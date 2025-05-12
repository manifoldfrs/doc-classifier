from __future__ import annotations

from importlib import import_module as _import_module

_pipeline = _import_module(".pipeline", package=__name__)

classify = _pipeline.classify
ClassificationResult = _pipeline.ClassificationResult

__all__: list[str] = [
    "classify",
    "ClassificationResult",
]
