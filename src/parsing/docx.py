from __future__ import annotations

import asyncio
import tempfile

import docx2txt
from starlette.datastructures import UploadFile


async def extract_text_from_docx(file: UploadFile) -> str:
    """
    Extract text content from a DOCX file using docx2txt.

    Args:
        file: The uploaded DOCX file

    Returns:
        Extracted text content as a string
    """
    await file.seek(0)
    content = await file.read()

    def _worker(docx_content: bytes) -> str:
        with tempfile.NamedTemporaryFile(suffix=".docx", delete=True) as temp_file:
            temp_file.write(docx_content)
            temp_file.flush()

            try:
                return docx2txt.process(temp_file.name) or ""
            except Exception as e:
                return f"Error extracting DOCX text: {str(e)}"

    return await asyncio.to_thread(_worker, content)
