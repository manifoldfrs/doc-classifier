"""src/classification/stages/text.py
###############################################################################
Text-content classification stage (Step 4.3)
###############################################################################
This module implements the *text* stage of the multi-stage document
classification pipeline.  It converts an uploaded file into **plain-text** using
one of the specialised extractors defined in :py:mod:`src.parsing.registry` and feeds the
resulting string into the machine-learning model exposed by
:pymod:`src.classification.model` when available.  The stage is designed to be
robust even when the model is not yet present (early implementation steps) by
falling back to *keyword heuristics* so that the pipeline continues to return
meaningful labels during incremental development.

Key characteristics
===================
1. **Async-friendly** – heavy synchronous parsing (e.g. pdfminer) happens inside
   each extractor via ``asyncio.to_thread`` so the event-loop remains
   responsive.  This function itself therefore remains non-blocking.
2. **Graceful degradation** – if ``src.classification.model`` cannot be
   imported *or* the model raises at runtime, the stage falls back to
   deterministic regex heuristics.
3. **≤ 40 lines policy** – the public coroutine **stage_text** is kept compact
   to respect the repository engineering rules.

Interface contract
------------------
``stage_text`` adheres to the *StageCallable* protocol defined in
:pymod:`src.classification.pipeline`, returning an immutable
:class:`~src.classification.pipeline.StageOutcome` instance.

Limitations / Future work
-------------------------
• The fallback heuristics are intentionally simple; they should be removed once
  the statistical model proves reliable.
• Language detection & non-English tokenisation are currently out-of-scope.
"""

from __future__ import annotations

# stdlib
import re
from typing import Dict, Pattern, Tuple

# third-party
import structlog
from starlette.datastructures import UploadFile

# local
from src.classification.pipeline import StageOutcome
from src.parsing.registry import TEXT_EXTRACTORS  # Updated import

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Optional model import – tolerated absence for early pipeline iterations
# ---------------------------------------------------------------------------
try:
    from src.classification import model as _model  # type: ignore

    _MODEL_AVAILABLE: bool = True
except ModuleNotFoundError:  # pragma: no cover – model implemented in Step 4.5
    _MODEL_AVAILABLE = False

# ---------------------------------------------------------------------------
# Heuristic keyword patterns used when the statistical model is not available
# ---------------------------------------------------------------------------
_LABEL_PATTERNS: Dict[str, Pattern[str]] = {
    "invoice": re.compile(r"\b(invoice|receipt|bill)\b", flags=re.I),
    "bank_statement": re.compile(r"\b(bank[_\- ]?statement|statement)\b", flags=re.I),
    "financial_report": re.compile(
        r"\b(annual[_\- ]?report|balance[_\- ]?sheet|financial[_\- ]?report)\b",
        flags=re.I,
    ),
    "drivers_licence": re.compile(r"\b(driver[s']?[_\- ]?licen[cs]e|dl)\b", flags=re.I),
    "contract": re.compile(r"\b(contract|agreement)\b", flags=re.I),
    "email": re.compile(r"\b(from:|to:|subject:)\b", flags=re.I),
    "form": re.compile(r"\b(application|form)\b", flags=re.I),
}

_FALLBACK_CONFIDENCE: float = 0.75  # deterministic score for heuristic path

# ---------------------------------------------------------------------------
# Helper – lightweight heuristic predictor (≤ 15 lines)
# ---------------------------------------------------------------------------


def _heuristic_predict(text: str) -> Tuple[str | None, float | None]:  # noqa: D401
    """Return *(label, confidence)* based on keyword heuristics.

    Parameters
    ----------
    text:
        Lower-cased document text.
    """

    for label, pattern in _LABEL_PATTERNS.items():
        if pattern.search(text):
            return label, _FALLBACK_CONFIDENCE
    return None, None


# ---------------------------------------------------------------------------
# Public stage callable (≤ 40 logical lines, excl. docstring & imports)
# ---------------------------------------------------------------------------


async def stage_text(file: UploadFile) -> StageOutcome:  # noqa: D401
    """Infer document type from **extracted textual content**.

    The function attempts to extract plaintext using an extension-specific
    parser.  If a statistical model is present it is used for classification;
    otherwise a regex fallback is applied.
    """

    if not file.filename or "." not in file.filename:
        return StageOutcome()

    ext: str = file.filename.rsplit(".", 1)[1].lower()
    extractor = TEXT_EXTRACTORS.get(ext)
    if extractor is None:
        return StageOutcome()

    # Extract text (the extractor is already async & off-threads heavy work)
    text: str = await extractor(file)
    if not text.strip():
        return StageOutcome()

    text_lower = text.lower()

    # ------------------------------------------------------------------
    # 1. Statistical model path – preferred
    # ------------------------------------------------------------------
    if _MODEL_AVAILABLE:
        try:
            label, confidence = _model.predict(text_lower)  # type: ignore[attr-defined]
            if label and confidence is not None:
                return StageOutcome(label=label, confidence=float(confidence))
        except (
            Exception
        ) as exc:  # noqa: BLE001 – specific errors unknown until Step 4.5
            logger.warning("text_stage_model_failure", error=str(exc))
            # fall through to heuristic path

    # ------------------------------------------------------------------
    # 2. Heuristic fallback path
    # ------------------------------------------------------------------
    label, confidence = _heuristic_predict(text_lower)
    if label and confidence is not None:
        return StageOutcome(label=label, confidence=confidence)

    return StageOutcome()
