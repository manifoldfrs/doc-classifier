from __future__ import annotations

from typing import Awaitable, Callable, Dict, Final

from starlette.datastructures import UploadFile

# Import _read_txt from the text stage module
from src.classification.stages.text import _read_txt

from .csv import extract_text_from_csv
from .docx import extract_text_from_docx
from .image import extract_text_from_image
from .pdf import extract_text_from_pdf

__all__: list[str] = [
    "TEXT_EXTRACTORS",
    "IMAGE_EXTRACTORS",
]

# Dispatch table – extension → coroutine for text-based files
TEXT_EXTRACTORS: Final[Dict[str, Callable[[UploadFile], Awaitable[str]]]] = {
    "pdf": extract_text_from_pdf,
    "docx": extract_text_from_docx,
    "csv": extract_text_from_csv,
    "txt": _read_txt,
}

# Dispatch table – extension → coroutine for image-based files (OCR)
IMAGE_EXTRACTORS: Final[Dict[str, Callable[[UploadFile], Awaitable[str]]]] = {
    "jpg": extract_text_from_image,
    "jpeg": extract_text_from_image,
    "png": extract_text_from_image,
    # "tiff": extract_text_from_image,
    # "bmp": extract_text_from_image,
}
