from __future__ import annotations

from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException
from starlette.datastructures import UploadFile

from src.ingestion.validators import validate_file
from tests.conftest import MockSettings


def _build_upload(
    filename: str | None,  # Allow None for testing
    payload: bytes,
    content_type: str | None = None,
) -> UploadFile:  # noqa: D401 – tiny factory
    """Return a Starlette *UploadFile* wrapping **payload** for isolation."""

    mock_file_obj = BytesIO(payload)

    # Create a proper MagicMock with the file attribute
    # The UploadFile constructor in Starlette takes filename, file, content_type
    # It's better to mock the UploadFile instance directly if we need to control its attributes deeply.
    upload_file_mock = MagicMock(spec=UploadFile)
    upload_file_mock.filename = filename
    upload_file_mock.file = mock_file_obj  # This is the SpooledTemporaryFile or BytesIO
    upload_file_mock.content_type = content_type

    # Mock async methods if needed by the calling code, though validate_file uses sync file ops
    upload_file_mock.seek = MagicMock()  # For the UploadFile.seek if it were async
    upload_file_mock.read = MagicMock(
        return_value=payload
    )  # For UploadFile.read if it were async

    # Ensure seek and tell are mocked on the underlying file object too
    # This is what _validate_size actually uses
    mock_file_obj.seek = MagicMock()
    mock_file_obj.tell = MagicMock()

    # Set a default return value for tell after seeking to end, to simulate size check
    def size_check_tell(*args):
        # Simulate returning current pos (0) then size
        if mock_file_obj.tell.call_count % 2 != 0:  # First call (get current)
            return 0
        else:  # Second call (get size)
            return len(payload)  # Return actual payload size

    mock_file_obj.tell.side_effect = size_check_tell

    return upload_file_mock


@pytest.fixture
def mock_settings() -> MockSettings:
    """Provides default mock settings for validation tests."""
    return MockSettings(
        max_file_size_mb=1, allowed_extensions_raw="txt,pdf,dat,weirdext"
    )


def test_validate_file_success_small_txt(mock_settings: MockSettings) -> None:
    """A well-formed text file within size & extension limits passes silently."""
    upload = _build_upload("hello.txt", b"hello world", "text/plain")
    validate_file(upload, settings=mock_settings)  # Should not raise


@pytest.mark.parametrize(
    "filename",
    [
        "malware.exe",  # unknown extension
        "archive.tar.gz",  # double extension with unsupported base
    ],
)
def test_unsupported_extension_raises(
    filename: str, mock_settings: MockSettings
) -> None:
    """Files with extensions outside the whitelist raise **415**."""
    # Modify settings for this specific test case if needed
    mock_settings.allowed_extensions = {"pdf", "txt"}  # Explicitly set allowed
    upload = _build_upload(filename, b"dummy", "application/octet-stream")

    with pytest.raises(HTTPException) as exc:
        validate_file(upload, settings=mock_settings)

    assert exc.value.status_code == 415
    assert "Unsupported file extension" in exc.value.detail


def test_file_too_large_raises(mock_settings: MockSettings) -> None:
    """Payload exceeding *MAX_FILE_SIZE_MB* triggers **413** entity-too-large."""
    mock_settings.max_file_size_mb = 1  # Set limit for test
    payload = b"z" * (2 * 1024 * 1024)  # 2MB > 1MB limit
    upload = _build_upload("big.pdf", payload, "application/pdf")

    # Re-configure mock tell for this specific payload size
    def size_check_tell_large(*args):
        if upload.file.tell.call_count % 2 != 0:
            return 0
        else:
            return len(payload)

    upload.file.tell.side_effect = size_check_tell_large

    with pytest.raises(HTTPException) as exc:
        validate_file(upload, settings=mock_settings)

    assert exc.value.status_code == 413
    assert "exceeds the limit of 1 MB" in exc.value.detail


def test_empty_file_raises(mock_settings: MockSettings) -> None:
    """Zero-byte upload yields **400** bad-request."""
    upload = _build_upload("empty.txt", b"", "text/plain")

    # Configure mock tell for zero size
    def size_check_tell_zero(*args):
        if upload.file.tell.call_count % 2 != 0:
            return 0
        else:
            return 0  # Size is 0

    upload.file.tell.side_effect = size_check_tell_zero

    with pytest.raises(HTTPException) as exc:
        validate_file(upload, settings=mock_settings)

    assert exc.value.status_code == 400
    assert exc.value.detail == "Uploaded file is empty."


def test_mime_type_mismatch_raises(mock_settings: MockSettings) -> None:
    """Mismatch between extension and MIME-type raises **415** unsupported-media."""
    upload = _build_upload(
        "doc.pdf", b"%PDF-1.4", "application/json"
    )  # pdf ext, json mime

    # Ensure mimetypes returns the expected type for the extension
    with patch("mimetypes.guess_type", return_value=("application/pdf", None)):
        with pytest.raises(HTTPException) as exc:
            validate_file(upload, settings=mock_settings)

        assert exc.value.status_code == 415
        assert (
            "MIME type mismatch: extension .pdf suggests application/pdf, but received application/json."
            in exc.value.detail
        )


def test_filename_missing_raises_400(mock_settings: MockSettings) -> None:
    """If file.filename is None or empty, it should raise 400."""
    upload_none_fn = _build_upload(None, b"some content", "text/plain")
    with pytest.raises(HTTPException) as exc_none:
        validate_file(upload_none_fn, settings=mock_settings)
    assert exc_none.value.status_code == 400
    assert exc_none.value.detail == "No filename provided."

    upload_empty_fn = _build_upload("", b"some content", "text/plain")
    with pytest.raises(HTTPException) as exc_empty:
        validate_file(upload_empty_fn, settings=mock_settings)
    assert exc_empty.value.status_code == 400
    assert exc_empty.value.detail == "No filename provided."


def test_file_has_no_extension_raises_415(mock_settings: MockSettings) -> None:
    """If a file has no extension (e.g., 'myfile'), it should raise 415."""
    upload = _build_upload(
        "myfilewithoutperiod", b"content", "application/octet-stream"
    )
    with pytest.raises(HTTPException) as exc:
        validate_file(upload, settings=mock_settings)
    assert exc.value.status_code == 415
    assert exc.value.detail == "File has no extension."


def test_file_size_at_limit_passes(mock_settings: MockSettings) -> None:
    """File size exactly at MAX_FILE_SIZE_MB should pass."""
    mock_settings.max_file_size_mb = 1  # Set limit
    one_mb_payload = b"a" * (1 * 1024 * 1024)
    upload = _build_upload("limitfile.dat", one_mb_payload, "application/octet-stream")

    # Configure mock tell for this specific size
    def size_check_tell_limit(*args):
        if upload.file.tell.call_count % 2 != 0:
            return 0
        else:
            return len(one_mb_payload)

    upload.file.tell.side_effect = size_check_tell_limit

    validate_file(upload, settings=mock_settings)  # Should not raise


def test_mime_type_validation_when_guess_is_none(mock_settings: MockSettings) -> None:
    """Test MIME validation when mimetypes.guess_type returns None for an extension."""
    # Assumes 'weirdext' is in allowed_extensions from fixture
    upload = _build_upload("file.weirdext", b"content", "application/x-custom")

    with patch("mimetypes.guess_type", return_value=(None, None)) as mock_guess:
        validate_file(upload, settings=mock_settings)  # Should not raise
        mock_guess.assert_called_once_with("file.weirdext")


def test_mime_type_validation_when_file_content_type_is_none(
    mock_settings: MockSettings,
) -> None:
    """Test MIME validation when UploadFile.content_type is None."""
    # Assumes 'txt' is in allowed_extensions from fixture
    upload = _build_upload("file.txt", b"content", content_type=None)

    with patch("mimetypes.guess_type", return_value=("text/plain", None)) as mock_guess:
        validate_file(upload, settings=mock_settings)  # Should not raise
        mock_guess.assert_not_called()


def test_seek_tell_oserror_handling(mock_settings: MockSettings) -> None:
    """Test that if file.file.seek or file.file.tell raises OSError, it's handled."""
    upload = _build_upload("test.pdf", b"some content", "application/pdf")

    # Mock the .file attribute's seek method to raise OSError
    upload.file.seek = MagicMock(side_effect=OSError("Simulated I/O error on seek"))

    with pytest.raises(HTTPException) as exc_info:
        validate_file(upload, settings=mock_settings)

    assert exc_info.value.status_code == 400
    assert "Unable to assess uploaded file size." in exc_info.value.detail

    # Reset mock and test for error on tell
    upload.file.seek = MagicMock()  # Reset seek to not raise error
    upload.file.tell = MagicMock(side_effect=OSError("Simulated I/O error on tell"))

    with pytest.raises(HTTPException) as exc_info_tell:
        validate_file(upload, settings=mock_settings)

    assert exc_info_tell.value.status_code == 400
    assert "Unable to assess uploaded file size." in exc_info_tell.value.detail
