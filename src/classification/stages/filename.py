"""src/classification/stages/filename.py
###############################################################################
Stage 1: Filename-based document classification
###############################################################################
This module implements the filename stage in the classification pipeline.
It analyzes the filename for patterns that suggest document types.
"""

from __future__ import annotations

import os
import re
from typing import Dict, Tuple

from starlette.datastructures import UploadFile

from src.classification.pipeline import StageOutcome

# Document patterns in filenames
# Maps regex patterns to (label, confidence) tuples
DOCUMENT_PATTERNS: Dict[str, Tuple[str, float]] = {
    r"invoice|inv\d+|bill": ("invoice", 0.85),
    r"bank.*state|statement": ("bank_statement", 0.85),
    r"financial.*report|report.*financial": ("financial_report", 0.85),
    r"driver.*licen[cs]e|licen[cs]e.*driver": ("drivers_licence", 0.85),
    r"id[_\s]?card|identity": ("id_doc", 0.85),
    r"contract|agreement|terms": ("contract", 0.85),
    r"email|e-mail|\.eml$": ("email", 0.85),
    r"form|application": ("form", 0.85),
}


async def stage_filename(file: UploadFile) -> StageOutcome:
    """
    Analyze filename to classify document type.

    Args:
        file: The uploaded file to analyze

    Returns:
        StageOutcome with label and confidence if recognized pattern found,
        otherwise label=None, confidence=None
    """
    if not file.filename:
        return StageOutcome(label=None, confidence=None)

    filename = os.path.basename(file.filename).lower()

    for pattern, (label, confidence) in DOCUMENT_PATTERNS.items():
        if re.search(pattern, filename, re.IGNORECASE):
            return StageOutcome(label=label, confidence=confidence)

    return StageOutcome(label=None, confidence=None)
