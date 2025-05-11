"""src/classification/stages/metadata.py
###############################################################################
Stage 2: Metadata-based document classification
###############################################################################
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

# stdlib
import asyncio
import re
from io import BytesIO
from typing import Dict, Pattern, Tuple

# third-party
import structlog
from pdfminer.high_level import extract_text
from starlette.datastructures import UploadFile

# local
from src.classification.pipeline import StageOutcome

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
METADATA_PATTERNS: Dict[Pattern, Tuple[str, float]] = {
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
    Extract PDF metadata from document content.

    Args:
        content: Raw PDF file content

    Returns:
        Extracted metadata as string
    """

    def _worker() -> str:
        try:
            # Use pdfminer to extract document info
            pdf_buffer = BytesIO(content)
            # Extract first page only for metadata
            return extract_text(pdf_buffer, page_numbers=[0], maxpages=1) or ""
        except Exception:
            return ""

    return await asyncio.to_thread(_worker)


async def stage_metadata(file: UploadFile) -> StageOutcome:
    """
    Analyze document metadata to classify document type.

    Currently only works with PDFs. Other file types are skipped.

    Args:
        file: The uploaded file to analyze

    Returns:
        StageOutcome with label and confidence if recognized pattern found,
        otherwise label=None, confidence=None
    """
    # Skip non-PDF files
    if not file.content_type or "pdf" not in file.content_type:
        return StageOutcome(label=None, confidence=None)

    # Extract metadata from PDF
    await file.seek(0)
    content = await file.read()
    metadata = await _extract_pdf_metadata(content)

    if not metadata:
        return StageOutcome(label=None, confidence=None)

    # Match metadata against patterns
    for pattern, (label, confidence) in METADATA_PATTERNS.items():
        if pattern.search(metadata):
            return StageOutcome(label=label, confidence=confidence)

    return StageOutcome(label=None, confidence=None)
