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

Parsing Strategy Registry
=========================
The actual dispatch tables (`TEXT_EXTRACTORS`, `IMAGE_EXTRACTORS`) are now
defined in :py:mod:`src.parsing.registry` to keep this `__init__.py` lean.
Classification stages should import the registry directly.
"""

from __future__ import annotations

# local – explicit, absolute imports keep mypy happy
from .csv import extract_text_from_csv
from .docx import extract_text_from_docx
from .image import extract_text_from_image
from .pdf import extract_text_from_pdf

# Re-export individual extractors for direct use if needed,
# but prefer using the registry in src.parsing.registry for stage dispatch.
__all__: list[str] = [
    "extract_text_from_pdf",
    "extract_text_from_docx",
    "extract_text_from_csv",
    "extract_text_from_image",
]
