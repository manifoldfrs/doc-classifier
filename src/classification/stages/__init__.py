"""src/classification/stages __init__
###############################################################################
Stage package initialiser – keeps imports lightweight.
###############################################################################
The subpackage groups individual *stage* implementations used by the
classification pipeline.  We avoid heavy side-effects at import-time; each
stage module is imported explicitly by :pymod:`src.classification.pipeline` so
that dependencies are only loaded when the pipeline starts.
"""

from __future__ import annotations

from importlib import import_module as _import_module

# Lazily import stages to avoid circulars during module graph construction.
_filename = _import_module(".filename", package=__name__)
_metadata = _import_module(".metadata", package=__name__)
_text = _import_module(".text", package=__name__)
_ocr = _import_module(".ocr", package=__name__)

stage_filename = _filename.stage_filename  # type: ignore[attr-defined]
stage_metadata = _metadata.stage_metadata  # type: ignore[attr-defined]
stage_text = _text.stage_text  # type: ignore[attr-defined]
stage_ocr = _ocr.stage_ocr  # type: ignore[attr-defined]

__all__: list[str] = [
    "stage_filename",
    "stage_metadata",
    "stage_text",
    "stage_ocr",
]
