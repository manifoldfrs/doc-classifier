"""
Stage 2: Metadata-based document classification

This module implements the metadata stage in the classification pipeline.
It extracts and analyzes document metadata (especially from PDFs) for
classification.

Why only PDFs?
==============
External libraries expose reliable cross-format metadata extraction only for
PDFs in the base stack (pdfminer.six).  Other formats (DOCX, images) require
additional packages (exifread, olefile) which are out-of-scope for the demo.
The stage therefore short-circuits when the uploaded file is **not** a PDF.

The stage runs entirely **off-thread** via :pyfunc:`asyncio.to_thread` so the
FastAPI event-loop remains responsive despite the synchronous pdfminer API.
"""

from __future__ import annotations

import asyncio
import re
from io import BytesIO
from typing import Dict, Pattern, Tuple

import structlog
from pdfminer.high_level import extract_text
from pdfminer.pdfdocument import PDFTextExtractionNotAllowed
from pdfminer.pdfparser import PDFSyntaxError
from pdfminer.pdftypes import PDFException
from pdfminer.psparser import PSException
from starlette.datastructures import UploadFile

from src.classification.types import StageOutcome

__all__: list[str] = ["stage_metadata"]

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Regex patterns applied to concatenated metadata values
# ---------------------------------------------------------------------------
_LABEL_PATTERNS: Dict[str, Pattern[str]] = {
    "invoice": re.compile(r"\b(invoice|receipt|bill)\b", flags=re.I),
    "bank_statement": re.compile(r"\b(bank statement|statement)\b", flags=re.I),
    "financial_report": re.compile(
        r"\b(financial report|annual report|balance sheet)\b", flags=re.I
    ),
    "drivers_licence": re.compile(r"\b(driver['s ]?licen[cs]e|dl)\b", flags=re.I),
    "contract": re.compile(r"\b(contract|agreement)\b", flags=re.I),
}

# Confidence is lower than filename stage because metadata can be noisy
_CONFIDENCE_SCORE: float = 0.86

# Document patterns in metadata
# Maps regex patterns to (label, confidence) tuples
METADATA_PATTERNS: Dict[Pattern[str], Tuple[str, float]] = {
    re.compile(r"invoice|receipt|bill", re.IGNORECASE): ("invoice", 0.86),
    re.compile(r"bank.*statement|statement", re.IGNORECASE): ("bank_statement", 0.83),
    re.compile(r"financial|report", re.IGNORECASE): ("financial_report", 0.82),
    re.compile(r"driver|licen[cs]e", re.IGNORECASE): ("drivers_licence", 0.88),
    re.compile(r"id.*card|identity|passport", re.IGNORECASE): ("id_doc", 0.84),
    re.compile(r"contract|agreement|terms", re.IGNORECASE): ("contract", 0.85),
    re.compile(r"email|e-mail", re.IGNORECASE): ("email", 0.87),
    re.compile(r"form|application", re.IGNORECASE): ("form", 0.81),
}


async def _extract_pdf_metadata(content: bytes) -> str:
    """
    Extract PDF metadata (approximated by first page text) from document content.

    Args:
        content: Raw PDF file content

    Returns:
        Extracted text from the first page as a proxy for metadata, or empty string on error.
    """

    def _worker() -> str:
        pdf_buffer = BytesIO(content)
        try:
            # Use pdfminer to extract first page text as metadata proxy
            return extract_text(pdf_buffer, page_numbers=[0], maxpages=1) or ""
        except PDFTextExtractionNotAllowed:
            logger.warning("pdf_metadata_extraction_denied")
            return ""
        except (PDFSyntaxError, PSException, PDFException) as e:
            logger.warning(
                "pdf_metadata_extraction_failed_pdfminer",
                error=str(e),
                error_type=type(e).__name__,
            )
            return ""
        # Removed broad `except Exception` - let unexpected errors propagate.

    # Run the synchronous pdfminer code in a separate thread
    return await asyncio.to_thread(_worker)


async def stage_metadata(file: UploadFile) -> StageOutcome:
    """
    Analyze document metadata (approximated by first page text) to classify type.

    Currently only processes PDF files. Other types are skipped.

    Args:
        file: The uploaded file to analyze

    Returns:
        StageOutcome with label and confidence if a recognized pattern is found
        in the metadata proxy, otherwise label=None, confidence=None.
    """
    # Skip non-PDF files, as metadata extraction is PDF-specific here
    if not file.content_type or "pdf" not in file.content_type.lower():
        logger.debug("metadata_stage_skip_non_pdf", filename=file.filename)
        return StageOutcome(label=None, confidence=None)

    try:
        # Read content once for metadata extraction
        await file.seek(0)
        content = await file.read()
        metadata_proxy = await _extract_pdf_metadata(content)
    except Exception as e:
        # Catch potential errors during file read/seek or the _extract_pdf_metadata call
        # This includes errors propagated from the _worker if they were not pdfminer-specific
        logger.error(
            "metadata_stage_processing_error",
            filename=file.filename,
            error=str(e),
            exc_info=True,  # Include traceback for unexpected errors
        )
        return StageOutcome(label=None, confidence=None)

    if not metadata_proxy or not metadata_proxy.strip():
        logger.debug("metadata_stage_no_metadata", filename=file.filename)
        return StageOutcome(label=None, confidence=None)

    # Match extracted metadata proxy against predefined patterns
    for pattern, (label, confidence) in METADATA_PATTERNS.items():
        if pattern.search(metadata_proxy):
            logger.debug(
                "metadata_stage_match",
                filename=file.filename,
                label=label,
                confidence=confidence,
            )
            return StageOutcome(label=label, confidence=confidence)

    logger.debug("metadata_stage_no_match", filename=file.filename)
    return StageOutcome(label=None, confidence=None)
