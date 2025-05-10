"""src/classification/stages/metadata.py
###############################################################################
Metadata-based heuristic classification stage (Step 4.2)
###############################################################################
This module implements the *metadata* stage of the document-classification
pipeline.  It inspects embedded metadata (currently PDF **DocumentInfo**)
searching for indicative keywords that map to domain-specific labels.

Why only PDFs?
==============
External libraries expose reliable cross-format metadata extraction only for
PDFs in the base stack (pdfminer.six).  Other formats (DOCX, images) require
additional packages (exifread, olefile) which are out-of-scope for the demo.
The stage therefore short-circuits when the uploaded file is **not** a PDF.

The stage runs entirely **off-thread** via :pyfunc:`asyncio.to_thread` so the
FastAPI event-loop remains responsive despite the synchronous pdfminer API.
"""

from __future__ import annotations

# stdlib
import asyncio
import re
from io import BytesIO
from typing import Dict, Pattern

# third-party
import structlog
from pdfminer.pdfdocument import PDFDocument  # type: ignore[import-not-found]
from pdfminer.pdfparser import PDFParser  # type: ignore[import-not-found]
from starlette.datastructures import UploadFile

# local
from src.classification.pipeline import StageOutcome

__all__: list[str] = ["stage_metadata"]

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Regex patterns applied to concatenated metadata values
# ---------------------------------------------------------------------------
_LABEL_PATTERNS: Dict[str, Pattern[str]] = {
    "invoice": re.compile(r"\b(invoice|receipt|bill)\b", flags=re.I),
    "bank_statement": re.compile(r"\b(bank statement|statement)\b", flags=re.I),
    "financial_report": re.compile(
        r"\b(financial report|annual report|balance sheet)\b", flags=re.I
    ),
    "drivers_licence": re.compile(r"\b(driver['s ]?licen[cs]e|dl)\b", flags=re.I),
    "contract": re.compile(r"\b(contract|agreement)\b", flags=re.I),
}

# Confidence is lower than filename stage because metadata can be noisy
_CONFIDENCE_SCORE: float = 0.86


async def _extract_pdf_metadata(pdf_bytes: bytes) -> str:
    """Return a **single string** containing all textual metadata values.

    The helper executes synchronously in a background thread to avoid blocking
    the event-loop.  When pdfminer raises an exception we return an empty
    string which causes the caller to yield an empty :class:`StageOutcome`.
    """

    def _worker() -> str:
        try:
            parser = PDFParser(BytesIO(pdf_bytes))
            document = PDFDocument(parser)
            if not document.info:
                return ""
            # `info` is a list of dicts – combine into one flat string
            parts = []
            for meta in document.info:
                for val in meta.values():
                    try:
                        parts.append(str(val))
                    except Exception:  # noqa: BLE001 narrow conversions vary
                        pass
            return " \n".join(parts)
        except Exception as exc:  # noqa: BLE001 pdfminer raises many types
            logger.warning("metadata_pdf_extraction_failed", error=str(exc))
            return ""

    return await asyncio.to_thread(_worker)


async def stage_metadata(file: UploadFile) -> StageOutcome:  # noqa: D401
    """Infer document label from **embedded metadata**.

    Only PDF uploads are processed.  Other formats return an empty outcome so
    downstream stages remain responsible for classification.
    """

    if not (file.content_type or "").lower().startswith("application/pdf"):
        logger.debug("metadata_stage_skipped", reason="non_pdf", mime=file.content_type)
        return StageOutcome()

    # Reset pointer & read bytes (≤10 MB → acceptable in memory)
    await file.seek(0)
    pdf_bytes: bytes = await file.read()

    meta_text: str = await _extract_pdf_metadata(pdf_bytes)
    if not meta_text:
        return StageOutcome()

    text_lower = meta_text.lower()
    for label, pattern in _LABEL_PATTERNS.items():
        if pattern.search(text_lower):
            logger.debug(
                "metadata_stage_match", label=label, confidence=_CONFIDENCE_SCORE
            )
            return StageOutcome(label=label, confidence=_CONFIDENCE_SCORE)

    logger.debug("metadata_stage_no_match", filename=file.filename)
    return StageOutcome()
