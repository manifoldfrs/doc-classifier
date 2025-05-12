"""
Stage 3: Text content-based document classification

This module implements the text stage in the classification pipeline.
It extracts text from supported document types (PDF, DOCX, CSV, TXT) and
analyzes the content using either a machine learning model (if available)
or heuristic patterns to determine the document type.

Key Responsibilities:
- Identify the correct text extractor based on file extension using the central registry.
- Extract text content from the file.
- Predict the document label using the ML model (primary method).
- Fall back to regex-based heuristics if the model is unavailable or fails.
- Return a StageOutcome with the determined label and confidence.

Dependencies:
- `src.parsing.registry`: Provides the central map of file extensions to text extractors.
- `src.classification.model`: Provides the `predict` function for ML-based classification.
- `src.classification.types`: Defines the `StageOutcome` data structure.
- `structlog`: Used for structured logging.
- `re`: Used for heuristic pattern matching.

"""

from __future__ import annotations

import re

import structlog
from starlette.datastructures import UploadFile

from src.classification.model import ModelNotAvailableError, predict
from src.classification.types import StageOutcome
from src.parsing.registry import TEXT_EXTRACTORS

logger = structlog.get_logger(__name__)

# Flag indicating whether ML model is intended to be used by this stage.
# Set to True to attempt model prediction first.
_MODEL_AVAILABLE = True


# Heuristic patterns for document classification (Fallback)
# These patterns are used if the ML model prediction is unavailable or fails.
# Maps regex patterns to (label, confidence) tuples. Confidence is typically
# lower than model confidence due to the simpler nature of heuristics.
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

    This stage attempts to extract text from the file based on its extension,
    using the extractors defined in `src.parsing.registry.TEXT_EXTRACTORS`.
    It then tries to classify the text using the ML model. If the model is not
    available or fails, it falls back to regex-based heuristic matching.

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

    # Check if a text extractor exists for this file type in the central registry
    if not extension or extension not in TEXT_EXTRACTORS:
        logger.debug(
            "text_stage_skip_unsupported_ext",
            filename=file.filename,
            extension=extension,
        )
        return StageOutcome(label=None, confidence=None)

    # Get the appropriate extractor function from the registry
    extractor = TEXT_EXTRACTORS[extension]
    try:
        # Ensure file pointer is reset before reading for extraction
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

    # If no text could be extracted, classification cannot proceed
    if not text or not text.strip():
        logger.debug("text_stage_skip_no_text", filename=file.filename)
        return StageOutcome(label=None, confidence=None)

    # Attempt classification using the ML model if configured
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
                # Return model prediction, rounding confidence
                return StageOutcome(
                    label=label, confidence=round(confidence, 4) if confidence else 0.0
                )
            # Log if model yielded no result, will fall through to heuristics
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
            # Fall through to heuristics if model artifact isn't found
        except Exception as e:
            logger.error(
                "text_stage_model_prediction_error",
                filename=file.filename,
                error=str(e),
                exc_info=True,
            )
            # If a generic error occurs during prediction, do not fallback.
            # Return unknown immediately to avoid potentially misleading heuristic match.
            return StageOutcome(label=None, confidence=None)

    # Fallback: Heuristic-based classification using regex patterns
    for pattern, (label, confidence) in _HEURISTIC_PATTERNS.items():
        if re.search(pattern, text, re.IGNORECASE):
            logger.debug(
                "text_stage_heuristic_match",
                filename=file.filename,
                label=label,
                confidence=confidence,
            )
            return StageOutcome(label=label, confidence=confidence)

    # If neither model nor heuristics produced a match
    logger.debug("text_stage_no_match", filename=file.filename, text_preview=text[:100])
    return StageOutcome(label=None, confidence=None)
