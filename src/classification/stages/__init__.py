from __future__ import annotations

from importlib import import_module as _import_module

# Lazily import stages to avoid circulars during module graph construction.
_filename = _import_module(".filename", package=__name__)
_metadata = _import_module(".metadata", package=__name__)
_text = _import_module(".text", package=__name__)
_ocr = _import_module(".ocr", package=__name__)

stage_filename = _filename.stage_filename
stage_metadata = _metadata.stage_metadata
stage_text = _text.stage_text
stage_ocr = _ocr.stage_ocr

__all__: list[str] = [
    "stage_filename",
    "stage_metadata",
    "stage_text",
    "stage_ocr",
]
