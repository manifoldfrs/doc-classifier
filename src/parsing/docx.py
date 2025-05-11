"""src/parsing/docx.py
###############################################################################
DOCX text extraction using docx2txt
###############################################################################
This module provides functions for extracting text from DOCX files.
"""

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
    # Read the file content
    await file.seek(0)
    content = await file.read()

    # Define worker function for async execution
    def _worker(docx_content: bytes) -> str:
        # Need to write to a temp file for docx2txt to process
        with tempfile.NamedTemporaryFile(suffix=".docx", delete=True) as temp_file:
            # Write content to temp file
            temp_file.write(docx_content)
            temp_file.flush()

            try:
                # Extract text
                return docx2txt.process(temp_file.name) or ""
            except Exception as e:
                # Handle docx extraction errors
                return f"Error extracting DOCX text: {str(e)}"

    # Run CPU-bound extraction in threadpool
    return await asyncio.to_thread(_worker, content)
