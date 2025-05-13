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
from typing import Dict, Optional, Pattern, Tuple

import structlog
from pdfminer.high_level import extract_text
from pdfminer.pdfdocument import PDFTextExtractionNotAllowed
from pdfminer.pdfparser import PDFSyntaxError
from pdfminer.pdftypes import PDFException
from pdfminer.psparser import PSException
from starlette.datastructures import UploadFile

from src.classification.types import StageOutcome
from src.core.exceptions import MetadataProcessingError

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


async def _extract_pdf_metadata(content: bytes, filename: Optional[str]) -> str:
    """
    Extract PDF metadata (approximated by first page text) from document content.

    Args:
        content: Raw PDF file content
        filename: The original filename (for logging context)

    Returns:
        Extracted text from the first page as a proxy for metadata, or empty string on error.
    """

    def _worker(pdf_content: bytes, worker_filename: Optional[str]) -> str:
        pdf_buffer = BytesIO(pdf_content)
        try:
            # Use pdfminer to extract first page text as metadata proxy
            return extract_text(pdf_buffer, page_numbers=[0], maxpages=1) or ""
        except PDFTextExtractionNotAllowed:
            logger.warning("pdf_metadata_extraction_denied", filename=worker_filename)
            return ""
        except (PDFSyntaxError, PSException, PDFException) as e:
            logger.warning(
                "pdf_metadata_extraction_failed_pdfminer",
                filename=worker_filename,  # Use passed filename
                error=str(e),
                error_type=type(e).__name__,
            )
            return ""
        except (
            Exception
        ) as e:  # Catch any other unexpected error during pdfminer processing
            logger.error(
                "pdf_metadata_extraction_unexpected_error",
                filename=worker_filename,  # Use passed filename
                error=str(e),
                exc_info=True,
            )
            # For truly unexpected errors in the worker, re-raise as MetadataProcessingError
            # This gives more context than a generic Exception to the caller.
            raise MetadataProcessingError(
                f"Unexpected error in PDF metadata worker: {str(e)}"
            ) from e

    # Run the synchronous pdfminer code in a separate thread, passing filename
    return await asyncio.to_thread(_worker, content, filename)


async def stage_metadata(file: UploadFile) -> StageOutcome:
    """
    Analyze document metadata (approximated by first page text) to classify type.

    Currently only processes PDF files. Other types are skipped.

    Args:
        file: The uploaded file to analyze

    Returns:
        StageOutcome with label and confidence if a recognized pattern is found
        in the metadata proxy, otherwise label=None, confidence=None.
    Raises:
        MetadataProcessingError: If a non-recoverable error occurs during processing.
    """
    # Skip non-PDF files, as metadata extraction is PDF-specific here
    if not file.content_type or "pdf" not in file.content_type.lower():
        logger.debug("metadata_stage_skip_non_pdf", filename=file.filename)
        return StageOutcome(label=None, confidence=None)

    try:
        # Read content once for metadata extraction
        await file.seek(0)
        content = await file.read()
        # Pass filename to the extraction helper
        metadata_proxy = await _extract_pdf_metadata(content, file.filename)
    except OSError as e:
        # Specific handling for I/O errors during file operations
        logger.error(
            "metadata_stage_io_error",
            filename=file.filename,
            error=str(e),
            exc_info=True,
        )
        # Wrap OSError in a domain-specific exception to be caught by pipeline
        raise MetadataProcessingError(
            f"File I/O error in metadata stage: {str(e)}"
        ) from e
    except MetadataProcessingError:
        # If _extract_pdf_metadata already raised our specific error, re-raise it
        raise
    except Exception as e:
        # Catch other potential errors during file read/seek or the _extract_pdf_metadata call
        logger.error(
            "metadata_stage_processing_error",
            filename=file.filename,
            error=str(e),
            exc_info=True,  # Include traceback for unexpected errors
        )
        # Wrap generic exceptions in MetadataProcessingError
        raise MetadataProcessingError(
            f"General processing error in metadata stage: {str(e)}"
        ) from e

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
