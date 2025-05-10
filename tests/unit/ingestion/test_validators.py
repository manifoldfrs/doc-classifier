"""tests/unit/ingestion/test_validators.py
###############################################################################
Unit tests for ``src.ingestion.validators.validate_file`` (Implementation Plan
– Step 9.2)
###############################################################################
The suite exercises the *critical* validation helper that guards every FastAPI
upload route.  We test **happy-path** acceptance plus four representative error
scenarios, asserting that the raised :class:`fastapi.HTTPException` carries the
expected *status_code* as mandated by the technical specification:

• 400 – empty file payload
• 413 – file exceeds configured MAX_FILE_SIZE_MB
• 415 – unsupported extension
• 415 – MIME-type mismatch

We avoid hitting the global *Settings* singleton by passing **custom instances**
with deterministic values so tests remain hermetic regardless of `.env` or CI
variables.
"""

from __future__ import annotations

# stdlib
from io import BytesIO

# third-party
import pytest
from fastapi import HTTPException
from starlette.datastructures import Headers, UploadFile

# local
from src.core.config import Settings
from src.ingestion.validators import validate_file

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_upload(
    filename: str,
    payload: bytes,
    content_type: str | None = None,
) -> UploadFile:  # noqa: D401 – tiny factory
    """Return a Starlette *UploadFile* wrapping **payload** for isolation."""

    buffer = BytesIO(payload)

    if content_type is not None:
        headers = Headers({"content-type": content_type})
    else:
        headers = None  # type: ignore[assignment]

    # Starlette 0.46+ signature: UploadFile(file, *, size=None, filename=None, headers=None)
    # Using keyword-only args ensures compatibility with older versions too –
    # they ignore unknown kwargs via **kwargs catch-all.
    return UploadFile(file=buffer, filename=filename, headers=headers)


# ---------------------------------------------------------------------------
# Tests – organised by behaviour class
# ---------------------------------------------------------------------------


def test_validate_file_success_small_txt() -> None:
    """A well-formed text file within size & extension limits passes silently."""

    settings = Settings(max_file_size_mb=1)  # generous limit for 11-byte payload
    upload = _build_upload("hello.txt", b"hello world", "text/plain")

    # Should *not* raise
    validate_file(upload, settings=settings)


@pytest.mark.parametrize(
    "filename",
    [
        "malware.exe",  # unknown extension
        "archive.tar.gz",  # double extension with unsupported base
    ],
)
def test_unsupported_extension_raises(filename: str) -> None:
    """Files with extensions outside the whitelist raise **415**."""

    settings = Settings()
    upload = _build_upload(filename, b"dummy", "application/octet-stream")

    with pytest.raises(HTTPException) as exc:
        validate_file(upload, settings=settings)

    assert exc.value.status_code == 415


def test_file_too_large_raises() -> None:
    """Payload exceeding *MAX_FILE_SIZE_MB* triggers **413** entity-too-large."""

    # Default MAX_FILE_SIZE_MB is 10 – craft a ~12 MB payload to exceed it
    payload = b"z" * (12 * 1024 * 1024)
    upload = _build_upload("big.pdf", payload, "application/pdf")

    with pytest.raises(HTTPException) as exc:
        validate_file(upload)  # use default Settings

    assert exc.value.status_code == 413


def test_empty_file_raises() -> None:
    """Zero-byte upload yields **400** bad-request."""

    settings = Settings()
    upload = _build_upload("empty.txt", b"", "text/plain")

    with pytest.raises(HTTPException) as exc:
        validate_file(upload, settings=settings)

    assert exc.value.status_code == 400


def test_mime_type_mismatch_raises() -> None:
    """Mismatch between extension and MIME-type raises **415** unsupported-media."""

    settings = Settings()
    # .pdf expected "application/pdf" but we fake JSON
    upload = _build_upload("doc.pdf", b"%PDF-1.4", "application/json")

    with pytest.raises(HTTPException) as exc:
        validate_file(upload, settings=settings)

    assert exc.value.status_code == 415
