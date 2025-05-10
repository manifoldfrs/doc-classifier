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

The module also provides the :pydata:`TEXT_EXTRACTORS` dispatch table mapping a
lower-case file extension (without leading dot) to its corresponding coroutine.
This enables the future pipeline orchestrator to perform a simple lookup
instead of multiple `if/elif` blocks.
"""

from __future__ import annotations

# stdlib
from typing import Awaitable, Callable, Dict, Final

# third-party
from starlette.datastructures import UploadFile

# local – explicit, absolute imports keep mypy happy
from .docx import extract_text_from_docx  # noqa: F401  (re-exported)
from .pdf import extract_text_from_pdf  # noqa: F401

__all__: list[str] = [
    "extract_text_from_pdf",
    "extract_text_from_docx",
    "TEXT_EXTRACTORS",
]

# ---------------------------------------------------------------------------
# Dispatch table – extension → coroutine
# ---------------------------------------------------------------------------
TEXT_EXTRACTORS: Final[Dict[str, Callable[[UploadFile], Awaitable[str]]]] = {
    "pdf": extract_text_from_pdf,
    "docx": extract_text_from_docx,
}
