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
from unittest.mock import MagicMock, patch

# third-party
import pytest
from fastapi import HTTPException
from starlette.datastructures import UploadFile

from src.ingestion.validators import validate_file

# local
from tests.conftest import MockSettings

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_upload(
    filename: str | None,  # Allow None for testing
    payload: bytes,
    content_type: str | None = None,
) -> UploadFile:  # noqa: D401 – tiny factory
    """Return a Starlette *UploadFile* wrapping **payload** for isolation."""

    # Create a proper MagicMock with the file attribute
    mock_file = MagicMock()
    mock_file.filename = filename

    # Set up a proper file attribute with seek/tell methods
    mock_file.file = BytesIO(payload)

    # Handle content_type
    if content_type is not None:
        mock_file.content_type = content_type
    else:
        mock_file.content_type = None

    # Create mock for async seek/read
    mock_file.seek = MagicMock()
    mock_file.read = MagicMock(return_value=payload)

    return mock_file  # type: ignore


# ---------------------------------------------------------------------------
# Tests – organised by behaviour class
# ---------------------------------------------------------------------------


def test_validate_file_success_small_txt() -> None:
    """A well-formed text file within size & extension limits passes silently."""

    settings = MockSettings(max_file_size_mb=1, allowed_extensions_raw="txt,pdf")
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

    settings = MockSettings(allowed_extensions_raw="pdf,txt")  # malware.exe not in here
    upload = _build_upload(filename, b"dummy", "application/octet-stream")

    with pytest.raises(HTTPException) as exc:
        validate_file(upload, settings=settings)

    assert exc.value.status_code == 415
    assert "Unsupported file extension" in exc.value.detail


def test_file_too_large_raises() -> None:
    """Payload exceeding *MAX_FILE_SIZE_MB* triggers **413** entity-too-large."""
    settings = MockSettings(max_file_size_mb=1, allowed_extensions_raw="pdf")
    payload = b"z" * (2 * 1024 * 1024)  # 2MB > 1MB limit
    upload = _build_upload("big.pdf", payload, "application/pdf")

    # Manually set up the size check behavior
    upload.file.seek = MagicMock()
    upload.file.tell = MagicMock(return_value=len(payload))  # Return the actual size

    with pytest.raises(HTTPException) as exc:
        validate_file(upload, settings=settings)

    assert exc.value.status_code == 413
    assert "exceeds the limit of 1 MB" in exc.value.detail


def test_empty_file_raises() -> None:
    """Zero-byte upload yields **400** bad-request."""

    settings = MockSettings(allowed_extensions_raw="txt")
    upload = _build_upload("empty.txt", b"", "text/plain")

    # Manually set up the size check behavior
    upload.file.seek = MagicMock()
    upload.file.tell = MagicMock(return_value=0)  # Empty file

    with pytest.raises(HTTPException) as exc:
        validate_file(upload, settings=settings)

    assert exc.value.status_code == 400
    assert exc.value.detail == "Uploaded file is empty."


def test_mime_type_mismatch_raises() -> None:
    """Mismatch between extension and MIME-type raises **415** unsupported-media."""
    settings = MockSettings(allowed_extensions_raw="pdf")
    # .pdf expected "application/pdf" but we fake JSON content_type
    upload = _build_upload("doc.pdf", b"%PDF-1.4", "application/json")

    # Mock the mimetypes.guess_type to return a predictable value
    with patch("mimetypes.guess_type", return_value=("application/pdf", None)):
        with pytest.raises(HTTPException) as exc:
            validate_file(upload, settings=settings)

        assert exc.value.status_code == 415
        assert "MIME type" in exc.value.detail


def test_filename_missing_raises_400() -> None:
    """If file.filename is None or empty, it should raise 400."""
    settings = MockSettings()

    # Test with None filename
    upload_none_fn = _build_upload(None, b"some content", "text/plain")
    # Ensure filename is truly None
    upload_none_fn.filename = None

    with pytest.raises(HTTPException) as exc_none:
        validate_file(upload_none_fn, settings=settings)
    assert exc_none.value.status_code == 400
    assert exc_none.value.detail == "No filename provided."

    # Test with empty string filename
    upload_empty_fn = _build_upload("", b"some content", "text/plain")
    with pytest.raises(HTTPException) as exc_empty:
        validate_file(upload_empty_fn, settings=settings)
    assert exc_empty.value.status_code == 400
    assert exc_empty.value.detail == "No filename provided."


def test_file_has_no_extension_raises_415() -> None:
    """If a file has no extension (e.g., 'myfile'), it should raise 415."""
    settings = MockSettings()
    upload = _build_upload(
        "myfilewithoutperiod", b"content", "application/octet-stream"
    )
    with pytest.raises(HTTPException) as exc:
        validate_file(upload, settings=settings)
    assert exc.value.status_code == 415
    assert exc.value.detail == "File has no extension."


def test_file_size_at_limit_passes() -> None:
    """File size exactly at MAX_FILE_SIZE_MB should pass."""
    settings = MockSettings(max_file_size_mb=1, allowed_extensions_raw="dat,pdf,txt")
    one_mb_payload = b"a" * (1 * 1024 * 1024)
    upload = _build_upload("limitfile.dat", one_mb_payload, "application/octet-stream")

    # Manually set up the size check behavior
    upload.file.seek = MagicMock()
    upload.file.tell = MagicMock(return_value=len(one_mb_payload))  # Exactly 1MB

    # Should not raise
    validate_file(upload, settings=settings)


def test_mime_type_validation_when_guess_is_none() -> None:
    """Test MIME validation when mimetypes.guess_type returns None for an extension."""
    settings = MockSettings(
        allowed_extensions_raw="weirdext"
    )  # Assume 'weirdext' is not in mimetypes

    upload = _build_upload("file.weirdext", b"content", "application/x-custom")

    # Mock mimetypes.guess_type to return None
    with patch("mimetypes.guess_type", return_value=(None, None)):
        # Should not raise MIME mismatch error
        validate_file(upload, settings=settings)


def test_mime_type_validation_when_file_content_type_is_none() -> None:
    """Test MIME validation when UploadFile.content_type is None."""
    settings = MockSettings(allowed_extensions_raw="txt")
    upload = _build_upload("file.txt", b"content", content_type=None)

    # Mock mimetypes.guess_type to return text/plain
    with patch("mimetypes.guess_type", return_value=("text/plain", None)):
        # Should not raise MIME mismatch error since file.content_type is None
        validate_file(upload, settings=settings)


def test_seek_tell_exception_handling() -> None:
    """Test that if file.seek or file.tell raises an exception, it's handled."""
    settings = MockSettings()

    # Create a proper mock with the right structure
    mock_upload_file = MagicMock(spec=UploadFile)
    mock_upload_file.filename = "test.pdf"
    mock_upload_file.content_type = "application/pdf"

    # Create a file attribute with methods that raise exceptions
    mock_file = MagicMock()
    mock_file.seek = MagicMock(side_effect=OSError("Simulated I/O error"))
    mock_file.tell = MagicMock(side_effect=OSError("Simulated I/O error"))

    # Attach the mock file to the mock upload file
    mock_upload_file.file = mock_file

    with pytest.raises(HTTPException) as exc_info:
        validate_file(mock_upload_file, settings=settings)

    assert exc_info.value.status_code == 400
    assert "Unable to assess uploaded file size." in exc_info.value.detail
