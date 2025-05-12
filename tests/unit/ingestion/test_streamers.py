from __future__ import annotations

import asyncio
from io import BytesIO
from typing import List

import pytest
from starlette.datastructures import UploadFile

from src.ingestion.streamers import DEFAULT_CHUNK_SIZE, stream_file


def _build_upload_file(content: bytes, filename: str = "test.bin") -> UploadFile:
    """Return an UploadFile wrapping **content** for test isolation.

    The *UploadFile* constructor signature across Starlette versions requires
    positional arguments.  Using keyword arguments for **content_type** causes
    a ``TypeError`` (`unexpected keyword argument`) under the fastapi/starlette
    versions pinned by our project.

    Positional invocation keeps compatibility regardless of upstream changes.
    """

    return UploadFile(
        filename=filename,
        file=BytesIO(content),
    )


def test_stream_exact_chunk_boundary() -> None:
    """When file size == chunk size the generator yields exactly one chunk."""

    payload: bytes = b"a" * DEFAULT_CHUNK_SIZE
    upload_file: UploadFile = _build_upload_file(payload)

    async def _collect() -> List[bytes]:
        chunks: List[bytes] = []
        async for chunk in stream_file(upload_file):
            chunks.append(chunk)
        return chunks

    chunks: List[bytes] = asyncio.run(_collect())

    assert chunks == [payload]


def test_stream_multiple_chunks_residue() -> None:
    """Generator yields N full-sized chunks plus a final residue chunk."""

    payload_size: int = (DEFAULT_CHUNK_SIZE * 2) + 17  # two full + residue
    payload: bytes = b"b" * payload_size
    upload_file: UploadFile = _build_upload_file(payload)

    async def _collect() -> List[bytes]:
        chunks: List[bytes] = []
        async for chunk in stream_file(upload_file):
            chunks.append(chunk)
        return chunks

    chunks: List[bytes] = asyncio.run(_collect())

    # ➤ 3 chunks expected: 64k, 64k, 17
    assert len(chunks) == 3
    assert len(chunks[-1]) == 17
    assert b"".join(chunks) == payload  # ordering & completeness


def test_invalid_chunk_size_raises() -> None:
    """Non-positive sizes must raise a ValueError immediately."""

    payload: bytes = b"irrelevant"
    upload_file: UploadFile = _build_upload_file(payload)

    async def _iter():
        async for _ in stream_file(
            upload_file, chunk_size=0
        ):  # noqa: PT017 generator must be consumed
            pass

    with pytest.raises(ValueError):
        asyncio.run(_iter())
