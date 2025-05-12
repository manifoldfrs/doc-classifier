"""
Parser Registry

This module centralizes the mapping between file extensions and their
corresponding text/image extraction functions (coroutines). The classification
stages use these registries to select the appropriate parser based on the
file type.

Key Responsibilities:
- Define and maintain mappings for text-based file extractors.
- Define and maintain mappings for image-based file extractors (OCR).

Dependencies:
- Individual parser modules (`.csv`, `.docx`, `.image`, `.pdf`, `.txt`).
- `starlette.datastructures.UploadFile`: Type hint for extractor functions.

"""

from __future__ import annotations

from typing import Awaitable, Callable, Dict, Final

from starlette.datastructures import UploadFile

from .csv import extract_text_from_csv
from .docx import extract_text_from_docx
from .image import extract_text_from_image
from .pdf import extract_text_from_pdf
from .txt import read_txt  # Import from the new dedicated txt parser module

__all__: list[str] = [
    "TEXT_EXTRACTORS",
    "IMAGE_EXTRACTORS",
]

# Dispatch table – maps lowercase file extensions to async text extraction functions.
TEXT_EXTRACTORS: Final[Dict[str, Callable[[UploadFile], Awaitable[str]]]] = {
    "pdf": extract_text_from_pdf,
    "docx": extract_text_from_docx,
    "csv": extract_text_from_csv,
    "txt": read_txt,
}

# Dispatch table – maps lowercase file extensions to async image OCR functions.
IMAGE_EXTRACTORS: Final[Dict[str, Callable[[UploadFile], Awaitable[str]]]] = {
    "jpg": extract_text_from_image,
    "jpeg": extract_text_from_image,
    "png": extract_text_from_image,
    # Additional image formats can be added here if needed, e.g.:
    # "tiff": extract_text_from_image,
    # "tif": extract_text_from_image,
    # "bmp": extract_text_from_image,
    # "gif": extract_text_from_image, # Note: GIF support in Tesseract/Pillow might vary
}
