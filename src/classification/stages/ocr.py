"""
Stage 4: OCR-based document classification

This module implements the OCR (Optical Character Recognition) stage in the
classification pipeline. It's primarily intended as a fallback mechanism for
image-based documents (e.g., JPG, PNG, TIFF) or scanned documents where text
extraction via other methods fails or is not applicable.

Key Responsibilities:
- Identify supported image file extensions using the central registry.
- Extract text from the image file using an OCR engine (Tesseract via pytesseract).
- Predict the document label using the ML model based on the extracted OCR text.
- Fall back to regex-based heuristics if the model is unavailable or fails.
- Return a StageOutcome with the determined label and confidence.

Dependencies:
- `src.parsing.registry`: Provides the central map of image extensions to OCR extractors.
- `src.classification.model`: Provides the `predict` function for ML-based classification.
- `src.classification.types`: Defines the `StageOutcome` data structure.
- `structlog`: Used for structured logging.
- `re`: Used for heuristic pattern matching.

Notes:
- OCR is computationally more expensive than direct text extraction.
- OCR accuracy depends heavily on image quality (resolution, clarity, skew).
"""

from __future__ import annotations

import re

import structlog
from starlette.datastructures import UploadFile

from src.classification.model import ModelNotAvailableError, predict
from src.classification.types import StageOutcome

# Import the central image extractor registry
from src.parsing.registry import IMAGE_EXTRACTORS

logger = structlog.get_logger(__name__)

# Flag indicating whether ML model is intended to be used by this stage.
# Set to True to attempt model prediction first.
_MODEL_AVAILABLE = True


# Heuristic patterns for OCR text classification (Fallback)
# These patterns are used if the ML model prediction is unavailable or fails.
# Confidence scores might be slightly lower than text stage heuristics due to
# potential OCR inaccuracies.
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

    This stage uses the OCR extractor defined in the central registry for
    supported image types. It then attempts classification using the ML model
    on the extracted text, falling back to heuristics if necessary.

    Args:
        file: The uploaded file object (`starlette.datastructures.UploadFile`).

    Returns:
        A `StageOutcome` containing the predicted label and confidence score,
        or (None, None) if classification is unsuccessful or the file type
        is unsupported by this stage.
    """
    # Determine file extension
    extension = None
    if file.filename:
        parts = file.filename.split(".")
        if len(parts) > 1:
            extension = parts[-1].lower()

    # Check if an OCR extractor exists for this file type in the central registry
    if not extension or extension not in IMAGE_EXTRACTORS:
        logger.debug(
            "ocr_stage_skip_unsupported_ext",
            filename=file.filename,
            extension=extension,
        )
        return StageOutcome(label=None, confidence=None)

    # Get the appropriate OCR extractor function from the registry
    extractor = IMAGE_EXTRACTORS[extension]
    try:
        # Ensure file pointer is reset before reading for OCR
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

    # If OCR extracted no text, classification cannot proceed
    if not text or not text.strip():
        logger.debug("ocr_stage_skip_no_text", filename=file.filename)
        return StageOutcome(label=None, confidence=None)

    # Attempt classification using the ML model if configured
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
                # Return model prediction, rounding confidence
                return StageOutcome(
                    label=label, confidence=round(confidence, 4) if confidence else 0.0
                )
            # Log if model yielded no result, will fall through to heuristics
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
            # Fall through to heuristics if model artifact isn't found
        except Exception as e:
            logger.error(
                "ocr_stage_model_prediction_error",
                filename=file.filename,
                error=str(e),
                exc_info=True,
            )
            # If a generic error occurs during prediction, do not fallback.
            # Return unknown immediately.
            return StageOutcome(label=None, confidence=None)

    # Fallback: Heuristic-based classification using regex patterns on OCR text
    for pattern, (label, confidence) in _HEURISTIC_PATTERNS.items():
        if re.search(pattern, text, re.IGNORECASE):
            logger.debug(
                "ocr_stage_heuristic_match",
                filename=file.filename,
                label=label,
                confidence=confidence,
            )
            return StageOutcome(label=label, confidence=confidence)

    # If neither model nor heuristics produced a match from OCR text
    logger.debug("ocr_stage_no_match", filename=file.filename, text_preview=text[:100])
    return StageOutcome(label=None, confidence=None)
