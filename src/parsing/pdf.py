from __future__ import annotations

import asyncio
from io import BytesIO

import structlog
from pdfminer.high_level import extract_text
from starlette.datastructures import UploadFile

__all__: list[str] = ["extract_text_from_pdf"]

logger = structlog.get_logger(__name__)


async def extract_text_from_pdf(file: UploadFile) -> str:
    """
    Extract text content from a PDF file using pdfminer.six.

    Args:
        file: The uploaded PDF file

    Returns:
        Extracted text content as a string
    """
    await file.seek(0)
    content = await file.read()

    def _worker(pdf_content: bytes) -> str:
        pdf_buffer = BytesIO(pdf_content)

        try:
            extracted_text = extract_text(pdf_buffer)
            return extracted_text or ""
        except Exception:
            return ""

    return await asyncio.to_thread(_worker, content)
