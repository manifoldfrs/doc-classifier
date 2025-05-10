"""src/ingestion/streamers.py
###############################################################################
Streamed file reader utilities used by the ingestion layer.

The *HeronAI* service never loads user-supplied files fully into memory.
Instead, we operate on **64 KB** chunks (`DEFAULT_CHUNK_SIZE`), yielding them
asynchronously so downstream classification stages can consume a non-blocking
byte stream even when processing large uploads.

Public API
==========
`stream_file()` – Asynchronous generator yielding `bytes` chunks from a
`starlette.datastructures.UploadFile` instance.  The helper guarantees:

1. A single seek to the file start so that previously inspected uploads (e.g.
   during validation) are rewinded before classification.
2. *Never* yields empty chunks – when `file.read()` returns an empty `bytes`
   object the generator exits gracefully.
3. Enforces a positive, non-zero `chunk_size` argument.

All logic is intentionally contained within a single function to abide by the
*single-responsibility* rule and the project's limit of **≤ 40 lines per
function**.

Edge cases & error handling
---------------------------
• Negative or zero `chunk_size` raises `ValueError` (programming error).
• Any unexpected I/O issues from the underlying `UploadFile` propagate
  naturally and are caught by the global FastAPI exception handler so we avoid
  swallowing stack traces.

The module does *not* close the file object – lifecycle management remains the
responsibility of the caller/FastAPI which disposes of the temporary upload
once the request completes.
###############################################################################
"""

from __future__ import annotations

# stdlib
from typing import AsyncGenerator, Final

# third-party
import structlog
from starlette.datastructures import UploadFile

__all__: list[str] = ["stream_file"]

logger = structlog.get_logger(__name__)

DEFAULT_CHUNK_SIZE: Final[int] = 64 * 1024  # 64 KB as mandated by spec


async def stream_file(
    file: UploadFile,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> AsyncGenerator[bytes, None]:
    """Yield a *stream* of ``bytes`` from **file** in *chunk_size* increments.

    Parameters
    ----------
    file:
        The :class:`starlette.datastructures.UploadFile` obtained from FastAPI
        route parameters.
    chunk_size:
        Number of bytes to read per iteration.  Must be **positive** and
        defaults to :pydata:`DEFAULT_CHUNK_SIZE` (64 KB).

    Yields
    ------
    bytes
        Raw binary chunks from the underlying temp-file until EOF.

    Raises
    ------
    ValueError
        If *chunk_size* is ≤ 0.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    # Ensure downstream consumers start at the beginning even if the file was
    # partially read during validation (e.g. MIME guess).  UploadFile exposes
    # an *async* seek which delegates to the underlying SpooledTemporaryFile.
    await file.seek(0)
    total_read: int = 0

    while True:
        chunk: bytes = await file.read(chunk_size)
        if not chunk:  # EOF reached – stop iteration
            break

        total_read += len(chunk)
        logger.debug(
            "stream_chunk_read",
            filename=file.filename,
            chunk_size=len(chunk),
            total_read=total_read,
        )
        yield chunk
