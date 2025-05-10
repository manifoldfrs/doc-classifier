"""src/parsing/pdf.py
###############################################################################
PDF text-extraction adapter
###############################################################################
The helper in this module converts a *binary* PDF payload obtained via
`starlette.datastructures.UploadFile` into a UTF-8 ``str`` suitable for the
classification pipeline.

Design-notes
============
• **Non-blocking** – `pdfminer.six` is CPU-bound and synchronous.  We therefore
  off-load processing to a background thread with :pyfunc:`asyncio.to_thread`.
• **Single responsibility** – exactly one public coroutine ≤ 40 lines of code.
• **No catch-all** – let `pdfminer` propagate domain-specific exceptions (e.g.
  `PDFSyntaxError`) so up-stack handlers can deal with them.  We *only* add a
  debug log entry on success for observability.
• **Memory considerations** – the PDF is already ≤ 10 MB (validated earlier).
  Reading it fully into memory is acceptable and cheaper than multiple I/O
  passes that `pdfminer` would otherwise perform on a temporary file.
"""

from __future__ import annotations

# stdlib
import asyncio
from io import BytesIO
from typing import Final

# third-party
import structlog
from pdfminer.high_level import extract_text  # type: ignore[import-not-found]
from starlette.datastructures import UploadFile

__all__: list[str] = ["extract_text_from_pdf"]

logger = structlog.get_logger(__name__)

# pdfminer default codec is latin-1; we always re-encode to UTF-8 for internal use
_PDFMINER_CODEC: Final[str] = "utf-8"


async def extract_text_from_pdf(file: UploadFile) -> str:  # noqa: D401
    """Return **plain-text** extracted from **PDF** *file*.

    Parameters
    ----------
    file:
        The *SpooledTemporaryFile*-backed upload provided by FastAPI.

    Returns
    -------
    str
        UTF-8 text representing the document contents.  The string may be empty
        when `pdfminer` fails to detect textual objects (e.g. scanned images);
        higher pipeline stages decide on OCR fallback.
    """

    # Ensure pointer at start in case previous validators have read from it.
    await file.seek(0)
    pdf_bytes: bytes = await file.read()

    # pdfminer expects a *file-like* object; wrap bytes in BytesIO.
    def _worker() -> str:
        return extract_text(BytesIO(pdf_bytes), codec=_PDFMINER_CODEC) or ""

    text: str = await asyncio.to_thread(_worker)
    logger.debug("pdf_text_extracted", filename=file.filename, characters=len(text))
    return text
