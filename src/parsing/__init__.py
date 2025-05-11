from __future__ import annotations

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
