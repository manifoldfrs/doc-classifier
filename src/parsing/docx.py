"""src/parsing/docx.py
###############################################################################
DOCX text-extraction adapter
###############################################################################
This adapter converts a binary **Microsoft Word (.docx)** document supplied as
`UploadFile` into plaintext.  We leverage :pymod:`docx2txt`, a lightweight
wrapper around the standard ``zipfile`` module that safely extracts XML
segments and concatenates the paragraph runs.

Unlike PDF, `docx2txt.process()` requires a *filesystem* path.  Creating a
short-lived temporary file is therefore unavoidable.  Given the 10 MB upload
limit this is acceptable and the OS performs the write on a memory-mapped tmpfs
in most container runtimes.
"""

from __future__ import annotations

# stdlib
import asyncio
import tempfile
from pathlib import Path

# third-party
import docx2txt  # type: ignore[import-not-found]
import structlog
from starlette.datastructures import UploadFile

__all__: list[str] = ["extract_text_from_docx"]

logger = structlog.get_logger(__name__)


async def extract_text_from_docx(file: UploadFile) -> str:  # noqa: D401
    """Return **plain-text** extracted from a `.docx` *file*.

    The coroutine writes the upload to a secure, namespaced temporary file that
    is deleted immediately after extraction.

    Parameters
    ----------
    file:
        The upload object obtained from FastAPI.

    Returns
    -------
    str
        Textual representation of the document.  May be empty for image-only
        Word documents.
    """

    await file.seek(0)
    docx_bytes: bytes = await file.read()

    # Write to NamedTemporaryFile so we avoid race-conditions on Windows.
    def _worker(tmp_path: Path) -> str:
        return docx2txt.process(str(tmp_path)) or ""

    with tempfile.NamedTemporaryFile(  # noqa: PTH123 safe within contextmanager
        suffix=".docx", delete=True
    ) as tmp:
        tmp.write(docx_bytes)
        tmp.flush()
        text: str = await asyncio.to_thread(_worker, Path(tmp.name))

    logger.debug("docx_text_extracted", filename=file.filename, characters=len(text))
    return text
