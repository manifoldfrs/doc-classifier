"""
Stage 3: Text content-based document classification

This module implements the text stage in the classification pipeline.
It analyzes document text content to determine document type.
"""

from __future__ import annotations

import re
from typing import Awaitable, Callable, Optional, Tuple

from starlette.datastructures import UploadFile

from src.classification.pipeline import StageOutcome
from src.parsing.csv import extract_text_from_csv
from src.parsing.docx import extract_text_from_docx
from src.parsing.pdf import extract_text_from_pdf

# Flag indicating whether ML model is available
_MODEL_AVAILABLE = False  # Set to True when ML model is implemented


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


class _MockModel:
    """Mock ML model for text classification until real model is implemented."""

    def predict(self, text: str) -> Tuple[Optional[str], float]:
        """Predict document type from text."""
        if not text or not text.strip():
            return None, 0.0

        # Use heuristic patterns as fallback
        for pattern, (label, confidence) in _HEURISTIC_PATTERNS.items():
            if re.search(pattern, text, re.IGNORECASE):
                return label, confidence

        return None, 0.0


# Initialize mock model
_model = _MockModel()


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
        extension = file.filename.split(".")[-1].lower()

    # Check if we have an extractor for this file type
    if not extension or extension not in TEXT_EXTRACTORS:
        return StageOutcome(label=None, confidence=None)

    # Extract text content
    extractor = TEXT_EXTRACTORS[extension]
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
