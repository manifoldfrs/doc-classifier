"""src/parsing/registry.py
###############################################################################
HeronAI ─ Parsing Strategy Registry
###############################################################################
This module defines the dispatch tables that map file extensions to their
corresponding text-extraction coroutines.  It centralises the registration of
parsing strategies, making it easier to manage and extend support for new file
types.

The dictionaries defined here are imported by the classification stages to
select the appropriate parser based on file extension.

Key Dictionaries
----------------
- TEXT_EXTRACTORS: Maps extensions of text-based files (PDF, DOCX, CSV) to
  their respective asynchronous extraction functions.
- IMAGE_EXTRACTORS: Maps extensions of image files (JPG, JPEG, PNG) to
  OCR-based asynchronous extraction functions.

Design Rationale
----------------
- Centralization: Provides a single place to manage parsing strategies.
- Clarity: Separates the definition of strategies from their usage.
- Extensibility: New parsers can be added by simply updating these
  dictionaries and ensuring the corresponding extractor functions are
  available.
"""

from __future__ import annotations

# stdlib
from typing import Awaitable, Callable, Dict, Final

# third-party
from starlette.datastructures import UploadFile

# local – explicit, absolute imports for clarity and mypy
from .csv import extract_text_from_csv
from .docx import extract_text_from_docx
from .image import extract_text_from_image
from .pdf import extract_text_from_pdf

__all__: list[str] = [
    "TEXT_EXTRACTORS",
    "IMAGE_EXTRACTORS",
]

# ---------------------------------------------------------------------------
# Dispatch table – extension → coroutine for text-based files
# ---------------------------------------------------------------------------
TEXT_EXTRACTORS: Final[Dict[str, Callable[[UploadFile], Awaitable[str]]]] = {
    "pdf": extract_text_from_pdf,
    "docx": extract_text_from_docx,
    "csv": extract_text_from_csv,
    # Add other text-based formats here, e.g.:
    # "txt": extract_text_from_txt,
    # "html": extract_text_from_html,
    # "xml": extract_text_from_xml,
    # "json": extract_text_from_json,
    # "md": extract_text_from_md,
    # "eml": extract_text_from_eml, (if a dedicated parser is created)
}

# ---------------------------------------------------------------------------
# Dispatch table – extension → coroutine for image-based files (OCR)
# ---------------------------------------------------------------------------
IMAGE_EXTRACTORS: Final[Dict[str, Callable[[UploadFile], Awaitable[str]]]] = {
    "jpg": extract_text_from_image,
    "jpeg": extract_text_from_image,
    "png": extract_text_from_image,
    # Add other image formats requiring OCR here, e.g.:
    # "tiff": extract_text_from_image,
    # "bmp": extract_text_from_image,
}
