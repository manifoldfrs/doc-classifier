"""src/classification/stages/filename.py
###############################################################################
Filename-based heuristic classification stage (Step 4.2)
###############################################################################
This module provides the *filename* stage of the multi-stage document
classification pipeline.  It inspects the uploaded filename for semantic
keywords (e.g. *invoice*, *bank_statement*).  When a keyword is detected the
stage returns a :class:`~src.classification.pipeline.StageOutcome` with an
associated confidence score, otherwise it yields an *empty* outcome so that
later stages (metadata, text, OCR) can attempt to infer the document type.
"""

from __future__ import annotations

# stdlib
import re
from typing import Dict, Pattern

# third-party
import structlog
from starlette.datastructures import UploadFile

# local
from src.classification.pipeline import StageOutcome

__all__: list[str] = ["stage_filename"]

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Pre-compiled regex patterns mapping → label
# ---------------------------------------------------------------------------
_LABEL_PATTERNS: Dict[str, Pattern[str]] = {
    "invoice": re.compile(r"\b(invoice|receipt|bill)\b", flags=re.I),
    "bank_statement": re.compile(r"\b(bank[_\- ]?statement|statement)\b", flags=re.I),
    "financial_report": re.compile(
        r"\b(annual[_\- ]?report|balance[_\- ]?sheet|financial[_\- ]?report)\b",
        flags=re.I,
    ),
    "drivers_licence": re.compile(r"\b(driver[s']?[_\- ]?licen[cs]e|dl)\b", flags=re.I),
    "id_doc": re.compile(r"\b(passport|id[_\- ]?card|identity)\b", flags=re.I),
    "contract": re.compile(r"\b(contract|agreement)\b", flags=re.I),
    "email": re.compile(r"\.(eml|msg)$", flags=re.I),  # extension only
    "form": re.compile(r"\b(form|application)\b", flags=re.I),
}

# Faster check for a keyword positioned at the start of the basename
_STRONG_START_REGEX = re.compile(r"^[a-z]+")


async def stage_filename(file: UploadFile) -> StageOutcome:  # noqa: D401
    """Infer document type based on the *filename* alone.

    Parameters
    ----------
    file:
        The :class:`~starlette.datastructures.UploadFile` received from the
        FastAPI route.

    Returns
    -------
    StageOutcome
        • ``label`` – The predicted document label or *None*.
        • ``confidence`` – A deterministic score in the range ``[0.80, 0.95]``
          when a match is found, otherwise *None*.
    """

    name: str = (file.filename or "").lower()
    if not name:
        logger.debug("filename_stage_no_filename")
        return StageOutcome()

    basename: str = name.rsplit("/", 1)[-1]  # remove any path component

    for label, pattern in _LABEL_PATTERNS.items():
        match = pattern.search(basename)
        if match:
            # Boost confidence when match occurs at very beginning of filename
            strong: bool = bool(
                _STRONG_START_REGEX.match(basename)
            ) and basename.startswith(match.group(0))
            confidence: float = 0.95 if strong else 0.80
            logger.debug(
                "filename_stage_match",
                label=label,
                confidence=confidence,
                filename=file.filename,
            )
            return StageOutcome(label=label, confidence=confidence)

    logger.debug("filename_stage_no_match", filename=file.filename)
    return StageOutcome()
