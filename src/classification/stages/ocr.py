"""
Stage 4: OCR-based document classification

This module implements the OCR stage in the classification pipeline.
It performs optical character recognition on images to extract text
for classification.
"""

from __future__ import annotations

import re

import structlog
from starlette.datastructures import UploadFile

from src.classification.model import ModelNotAvailableError, predict
from src.classification.types import StageOutcome
from src.parsing.image import extract_text_from_image

logger = structlog.get_logger(__name__)

# Flag indicating whether ML model is available
_MODEL_AVAILABLE = True

# Mapping of file extensions to image extractors
IMAGE_EXTRACTORS = {
    "jpg": extract_text_from_image,
    "jpeg": extract_text_from_image,
    "png": extract_text_from_image,
    "tiff": extract_text_from_image,
    "tif": extract_text_from_image,
    "bmp": extract_text_from_image,
    "gif": extract_text_from_image,
}

# Heuristic patterns for OCR text classification
_HEURISTIC_PATTERNS = {
    r"invoice|receipt|bill|amount\s+due|payment|total\s+due": ("invoice", 0.72),
    r"bank|statement|account.*balance|withdrawal|deposit": ("bank_statement", 0.72),
    r"financial|report|quarterly|annual|balance\s+sheet": ("financial_report", 0.72),
    r"driver|licen[cs]e|dmv|permit|vehicle": ("drivers_licence", 0.72),
    r"passport|identity|id\s+card|identification": ("id_doc", 0.72),
    r"contract|agreement|terms|parties|hereby\s+agree": ("contract", 0.72),
    r"application|form|please\s+complete|applicant": ("form", 0.72),
}


async def stage_ocr(file: UploadFile) -> StageOutcome:
    """
    Perform OCR on image files to extract and classify text.

    This stage is typically a fallback for when other stages couldn't
    classify the document, especially for scanned documents.

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
    if not extension or extension not in IMAGE_EXTRACTORS:
        logger.debug(
            "ocr_stage_skip_unsupported_ext",
            filename=file.filename,
            extension=extension,
        )
        return StageOutcome(label=None, confidence=None)

    # Extract text through OCR
    extractor = IMAGE_EXTRACTORS[extension]
    try:
        # Ensure file pointer is reset before reading
        await file.seek(0)
        text = await extractor(file)
    except Exception as e:
        logger.error(
            "ocr_stage_extraction_error",
            filename=file.filename,
            extension=extension,
            error=str(e),
            exc_info=True,
        )
        return StageOutcome(label=None, confidence=None)

    # Skip if no text extracted
    if not text or not text.strip():
        logger.debug("ocr_stage_skip_no_text", filename=file.filename)
        return StageOutcome(label=None, confidence=None)

    # Use ML model if available, otherwise fallback to heuristics
    if _MODEL_AVAILABLE:
        try:
            label, confidence = predict(text)
            if label and confidence is not None:
                logger.debug(
                    "ocr_stage_model_prediction",
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
                "ocr_stage_model_no_prediction",
                filename=file.filename,
                text_preview=text[:100],
            )
        except ModelNotAvailableError:
            logger.warning(
                "ocr_stage_model_not_available",
                filename=file.filename,
                fallback="heuristics",
            )
            # Fall through to heuristics
        except Exception as e:
            logger.error(
                "ocr_stage_model_prediction_error",
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
                "ocr_stage_heuristic_match",
                filename=file.filename,
                label=label,
                confidence=confidence,
            )
            return StageOutcome(label=label, confidence=confidence)

    logger.debug("ocr_stage_no_match", filename=file.filename, text_preview=text[:100])
    return StageOutcome(label=None, confidence=None)
