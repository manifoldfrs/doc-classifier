"""src/parsing/__init__.py
###############################################################################
HeronAI ─ Parsing Package Root
###############################################################################
Central interface for *file-type specific* text-extraction helpers.

Public helpers
--------------
The package exposes **asynchronous** helper functions that convert binary
uploads into plaintext strings.  Each function is specialised for a specific
mime/extension and MUST:

1. Never block the event-loop – CPU-bound work is off-loaded via
   `asyncio.to_thread()` so the FastAPI worker remains responsive.
2. Raise *type-specific* exceptions – callers are expected to know which parser
   they invoke; global error handling happens higher up the stack.
3. Return *raw* text; they do **not** perform cleaning, deduplication, or any
   ML-specific tokenisation.  Those concerns belong in the classification
   stages.

Dispatch tables
===============
The module also provides two look-up dictionaries for downstream orchestration:

* :pydata:`TEXT_EXTRACTORS` – maps *text-friendly* formats directly parseable
  without OCR (PDF, DOCX, CSV) to their async extractor.
* :pydata:`IMAGE_EXTRACTORS` – maps raster image formats to OCR-based
  extractors.  The classification pipeline will consult this table when
  falling back to OCR.
"""

from __future__ import annotations

# stdlib
from typing import Awaitable, Callable, Dict, Final

# third-party
from starlette.datastructures import UploadFile

# local – explicit, absolute imports keep mypy happy
from .csv import extract_text_from_csv  # noqa: F401
from .docx import extract_text_from_docx  # noqa: F401  (re-exported)
from .image import extract_text_from_image  # noqa: F401
from .pdf import extract_text_from_pdf  # noqa: F401

__all__: list[str] = [
    "extract_text_from_pdf",
    "extract_text_from_docx",
    "extract_text_from_csv",
    "extract_text_from_image",
    "TEXT_EXTRACTORS",
    "IMAGE_EXTRACTORS",
]

# ---------------------------------------------------------------------------
# Dispatch table – extension → coroutine
# ---------------------------------------------------------------------------
TEXT_EXTRACTORS: Final[Dict[str, Callable[[UploadFile], Awaitable[str]]]] = {
    "pdf": extract_text_from_pdf,
    "docx": extract_text_from_docx,
    "csv": extract_text_from_csv,
}

# Raster images rely on OCR – defined separately for clarity
IMAGE_EXTRACTORS: Final[Dict[str, Callable[[UploadFile], Awaitable[str]]]] = {
    "jpg": extract_text_from_image,
    "jpeg": extract_text_from_image,
    "png": extract_text_from_image,
}
