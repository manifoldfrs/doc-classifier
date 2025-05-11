"""src/classification/stages/ocr.py
###############################################################################
Stage 4: OCR-based document classification
###############################################################################
This module implements the OCR stage in the classification pipeline.
It performs optical character recognition on images to extract text
for classification.
"""

from __future__ import annotations

import re
from typing import Tuple

from starlette.datastructures import UploadFile

from src.classification.pipeline import StageOutcome
from src.parsing.image import extract_text_from_image

# Flag indicating whether ML model is available
_MODEL_AVAILABLE = False  # Set to True when ML model is implemented

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


class _MockModel:
    """Mock ML model for OCR text classification until real model is implemented."""

    def predict(self, text: str) -> Tuple[str, float]:
        """Predict document type from OCR text."""
        if not text or not text.strip():
            return None, 0.0

        # Use heuristic patterns as fallback
        for pattern, (label, confidence) in _HEURISTIC_PATTERNS.items():
            if re.search(pattern, text, re.IGNORECASE):
                return label, confidence

        return None, 0.0


# Initialize mock model
_model = _MockModel()


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
        extension = file.filename.split(".")[-1].lower()

    # Check if we have an extractor for this file type
    if not extension or extension not in IMAGE_EXTRACTORS:
        return StageOutcome(label=None, confidence=None)

    # Extract text through OCR
    extractor = IMAGE_EXTRACTORS[extension]
    text = await extractor(file)

    # Skip if no text extracted
    if not text or not text.strip():
        return StageOutcome(label=None, confidence=None)

    # Use ML model if available, otherwise fallback to heuristics
    if _MODEL_AVAILABLE:
        label, confidence = _model.predict(text)
        if label:
            return StageOutcome(label=label, confidence=confidence)
    else:
        # Fallback to heuristic classification
        for pattern, (label, confidence) in _HEURISTIC_PATTERNS.items():
            if re.search(pattern, text, re.IGNORECASE):
                return StageOutcome(label=label, confidence=confidence)

    return StageOutcome(label=None, confidence=None)
