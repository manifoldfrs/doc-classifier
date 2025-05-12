"""
Stage 3: Text content-based document classification

This module implements the text stage in the classification pipeline.
It analyzes document text content to determine document type.
"""

from __future__ import annotations

import re
from typing import Awaitable, Callable

import structlog
from starlette.datastructures import UploadFile

from src.classification.model import ModelNotAvailableError, predict
from src.classification.types import StageOutcome
from src.parsing.csv import extract_text_from_csv
from src.parsing.docx import extract_text_from_docx
from src.parsing.pdf import extract_text_from_pdf

logger = structlog.get_logger(__name__)

# Flag indicating whether ML model is available
_MODEL_AVAILABLE = True


# Text extractors mapping file extensions to extraction functions
async def _read_txt(file: UploadFile) -> str:  # noqa: D401 – tiny helper
    """Read plain-text files fully (runs off-thread implicitly via Starlette)."""

    await file.seek(0)
    data = await file.read()
    return data.decode("utf-8", errors="replace")


TEXT_EXTRACTORS: dict[str, Callable[[UploadFile], Awaitable[str]]] = {
    "pdf": extract_text_from_pdf,
    "docx": extract_text_from_docx,
    "csv": extract_text_from_csv,
    "txt": _read_txt,
}

# Heuristic patterns for document classification
_HEURISTIC_PATTERNS = {
    r"invoice|receipt|bill|amount\s+due|payment|total\s+due": ("invoice", 0.75),
    r"bank|statement|account.*balance|withdrawal|deposit": ("bank_statement", 0.75),
    r"financial|report|quarterly|annual|balance\s+sheet": ("financial_report", 0.75),
    r"driver|licen[cs]e|dmv|permit|vehicle": ("drivers_licence", 0.75),
    r"passport|identity|id\s+card|identification": ("id_doc", 0.75),
    r"contract|agreement|terms|parties|hereby\s+agree": ("contract", 0.75),
    r"application|form|please\s+complete|applicant": ("form", 0.75),
}


async def stage_text(file: UploadFile) -> StageOutcome:
    """
    Analyze document text content to determine document type.

    Args:
        file: The uploaded file to analyze

    Returns:
        StageOutcome with label and confidence if document type identified,
        otherwise label=None, confidence=None
    """
    # Get file extension
    extension = None
    if file.filename:
        parts = file.filename.split(".")
        if len(parts) > 1:
            extension = parts[-1].lower()

    # Check if we have an extractor for this file type
    if not extension or extension not in TEXT_EXTRACTORS:
        logger.debug(
            "text_stage_skip_unsupported_ext",
            filename=file.filename,
            extension=extension,
        )
        return StageOutcome(label=None, confidence=None)

    # Extract text content
    extractor = TEXT_EXTRACTORS[extension]
    try:
        # Ensure file pointer is reset before reading
        await file.seek(0)
        text = await extractor(file)
    except Exception as e:
        logger.error(
            "text_stage_extraction_error",
            filename=file.filename,
            extension=extension,
            error=str(e),
            exc_info=True,
        )
        return StageOutcome(label=None, confidence=None)

    # Skip if no text extracted
    if not text or not text.strip():
        logger.debug("text_stage_skip_no_text", filename=file.filename)
        return StageOutcome(label=None, confidence=None)

    # Use ML model if available, otherwise fallback to heuristics
    if _MODEL_AVAILABLE:
        try:
            label, confidence = predict(text)
            if label and confidence is not None:
                logger.debug(
                    "text_stage_model_prediction",
                    filename=file.filename,
                    label=label,
                    confidence=confidence,
                )
                # Return model prediction only if confidence is meaningful
                return StageOutcome(
                    label=label, confidence=round(confidence, 4) if confidence else 0.0
                )
            # Fall through to heuristics if model provides no label or confidence
            logger.debug(
                "text_stage_model_no_prediction",
                filename=file.filename,
                text_preview=text[:100],
            )
        except ModelNotAvailableError:
            logger.warning(
                "text_stage_model_not_available",
                filename=file.filename,
                fallback="heuristics",
            )
            # Fall through to heuristics
        except Exception as e:
            logger.error(
                "text_stage_model_prediction_error",
                filename=file.filename,
                error=str(e),
                exc_info=True,
            )
            # Don't fallback on generic model error, return unknown
            return StageOutcome(label=None, confidence=None)

    # Fallback to heuristic classification
    for pattern, (label, confidence) in _HEURISTIC_PATTERNS.items():
        if re.search(pattern, text, re.IGNORECASE):
            logger.debug(
                "text_stage_heuristic_match",
                filename=file.filename,
                label=label,
                confidence=confidence,
            )
            return StageOutcome(label=label, confidence=confidence)

    logger.debug("text_stage_no_match", filename=file.filename, text_preview=text[:100])
    return StageOutcome(label=None, confidence=None)
