"""src/classification/stages/ocr.py
###############################################################################
OCR-based classification stage (Step 4.3)
###############################################################################
This module implements the *OCR* stage of the document-classification pipeline.
It is executed **after** the preceding text stage when the uploaded file is a
*bitmap image* (JPEG/PNG) or any format for which the text extractor yielded
insufficient content.  Raster images are converted to plaintext via Tesseract
(see :pymod:`src.parsing.image`).  The recognised text is then classified using
``src.classification.model`` if available, mirroring the logic used in the
*text* stage.

Design highlights
=================
1. **Extension gate** – the stage only triggers for extensions registered in
   :pydata:`src.parsing.registry.IMAGE_EXTRACTORS` ensuring we do not waste CPU cycles on
   PDFs/DOCX that are better handled elsewhere.
2. **Shared heuristics** – in the absence of a statistical model we reuse the
   same regex patterns defined in ``text.py`` for deterministic fallback.
3. **≤ 40 lines** – the public coroutine respects repository rules.
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
from src.parsing.registry import IMAGE_EXTRACTORS  # Updated import

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Optional model import – remain functional when model not yet implemented
# ---------------------------------------------------------------------------
try:
    from src.classification import model as _model  # type: ignore

    _MODEL_AVAILABLE: bool = True
except ModuleNotFoundError:  # pragma: no cover
    _MODEL_AVAILABLE = False

# ---------------------------------------------------------------------------
# Heuristic patterns (duplicated to avoid cross-module import cycles)
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

_FALLBACK_CONFIDENCE: float = 0.72  # slightly lower than text stage


def _heuristic_predict(text: str) -> Tuple[str | None, float | None]:  # noqa: D401
    """Simple keyword-based predictor used when the ML model is absent."""

    for label, pattern in _LABEL_PATTERNS.items():
        if pattern.search(text):
            return label, _FALLBACK_CONFIDENCE
    return None, None


# ---------------------------------------------------------------------------
# Public stage callable
# ---------------------------------------------------------------------------


async def stage_ocr(file: UploadFile) -> StageOutcome:  # noqa: D401
    """Infer label via **OCR** for raster images.

    Parameters
    ----------
    file:
        The image uploaded by the client.
    """

    if not file.filename or "." not in file.filename:
        return StageOutcome()

    ext: str = file.filename.rsplit(".", 1)[1].lower()
    extractor = IMAGE_EXTRACTORS.get(ext)
    if extractor is None:
        return StageOutcome()

    text: str = await extractor(file)
    if not text.strip():
        return StageOutcome()

    text_lower = text.lower()

    # Preferred path – statistical model
    if _MODEL_AVAILABLE:
        try:
            label, confidence = _model.predict(text_lower)  # type: ignore[attr-defined]
            if label and confidence is not None:
                return StageOutcome(label=label, confidence=float(confidence))
        except Exception as exc:  # noqa: BLE001 – model errors unknown
            logger.warning("ocr_stage_model_failure", error=str(exc))
            # fall through to heuristic path

    label, confidence = _heuristic_predict(text_lower)
    if label and confidence is not None:
        return StageOutcome(label=label, confidence=confidence)

    return StageOutcome()
