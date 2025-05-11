from __future__ import annotations

from typing import AsyncGenerator, Final

import structlog
from starlette.datastructures import UploadFile

__all__: list[str] = ["stream_file"]

logger = structlog.get_logger(__name__)

DEFAULT_CHUNK_SIZE: Final[int] = 64 * 1024  # 64 KB


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
        if not chunk:
            break

        total_read += len(chunk)
        logger.debug(
            "stream_chunk_read",
            filename=file.filename,
            chunk_size=len(chunk),
            total_read=total_read,
        )
        yield chunk
