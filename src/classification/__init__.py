from __future__ import annotations

from importlib import import_module as _import_module

_pipeline_module = _import_module(".pipeline", package=__name__)
classify = _pipeline_module.classify

_types_module = _import_module(".types", package=__name__)
ClassificationResult = _types_module.ClassificationResult
StageOutcome = _types_module.StageOutcome

__all__: list[str] = [
    "classify",
    "ClassificationResult",
    "StageOutcome",
]
