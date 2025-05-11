###############################################################################
# src/ingestion/validators.py
# -----------------------------------------------------------------------------
# Validation utilities for *incoming* user uploads.  All FastAPI routes dealing
# with `UploadFile` objects should call `validate_file()` *before* performing
# any further processing so that:
#
# 1. Unsupported or malicious files are rejected early (fail-fast principle).
# 2. Downstream code can rely on guaranteed invariants – non-empty, size ≤ limit,
#    extension & MIME type whitelisted.
#
# The function raises `fastapi.HTTPException` using **semantic** HTTP status
# codes as mandated by the technical specification:
#   • 400 – generic validation failure (empty filename / empty content).
#   • 413 – payload too large (> MAX_FILE_SIZE_MB).
#   • 415 – unsupported media type (extension or MIME mismatch).
#
# No *catch-all* except blocks are used – we explicitly handle known failure
# modes and let genuinely unexpected errors propagate to the global FastAPI
# exception handlers.
###############################################################################

from __future__ import annotations

# stdlib
import mimetypes
import os
from typing import Final, Optional

# third-party
import structlog
from fastapi import HTTPException, UploadFile

# local
from src.core.config import Settings, get_settings

__all__: list[str] = ["validate_file"]

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants & helpers
# ---------------------------------------------------------------------------
# The Starlette `UploadFile` object defers actual disk writes until the file
# exceeds a size threshold.  Seeking is therefore safe and *does not* force the
# file into memory – the underlying SpooledTemporaryFile handles buffering.
#
# We perform a *single* seek/tell/seek cycle to determine the file size; this is
# O(1) and avoids reading the entire payload.
_SIZE_SEEK_END: Final[int] = 2  # whence value for file.seek(..., 2)
_SIZE_SEEK_START: Final[int] = 0  # whence value for file.seek(..., 0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def validate_file(file: UploadFile, *, settings: Optional[Settings] = None) -> None:
    """
    Validate an uploaded file against configured restrictions.

    Performs checks for:
    - Empty files
    - File size limits
    - Allowed extensions
    - MIME type consistency

    Args:
        file: The uploaded file to validate
        settings: Optional Settings instance (uses global if not provided)

    Raises:
        HTTPException: With appropriate status code if validation fails
            - 400: Empty file or missing filename
            - 413: File too large
            - 415: Unsupported extension or MIME type mismatch
    """
    settings = settings or get_settings()

    # Check if filename exists
    if not file.filename:
        logger.warning("file_upload_no_filename")
        raise HTTPException(status_code=400, detail="No filename provided.")

    # Extract and validate extension
    filename = file.filename.lower()
    _, extension = os.path.splitext(filename)

    if not extension:
        logger.warning("file_upload_no_extension", filename=filename)
        raise HTTPException(status_code=415, detail="File has no extension.")

    # Remove the dot from extension
    extension = extension[1:]

    # Check if extension is allowed
    if not settings.is_extension_allowed(extension):
        logger.warning("file_upload_invalid_extension", extension=extension)
        raise HTTPException(
            status_code=415, detail=f"Unsupported file extension: .{extension}"
        )

    # Check file size
    try:
        file.file.seek(0, 2)  # Seek to end
        size = file.file.tell()
        file.file.seek(0)  # Reset to beginning

        # Check if file is empty
        if size == 0:
            logger.warning("file_upload_empty", filename=filename)
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        # Check if file is too large
        max_size = settings.max_file_size_mb * 1024 * 1024  # Convert MB to bytes
        if size > max_size:
            logger.warning(
                "file_upload_too_large",
                size=size,
                max_size=max_size,
                filename=filename,
            )
            raise HTTPException(
                status_code=413,
                detail=f"File size {size/1024/1024:.2f} MB exceeds the limit of {settings.max_file_size_mb} MB.",
            )
    except OSError as e:
        logger.error("file_size_check_failed", error=str(e), filename=filename)
        raise HTTPException(
            status_code=400,
            detail="Unable to assess uploaded file size. The upload may be corrupted.",
        ) from e

    # Validate MIME type consistency if both content_type and extension are available
    if file.content_type and extension:
        expected_mime = mimetypes.guess_type(f"file.{extension}")[0]

        # Only check if we have a guess for this extension
        if expected_mime and file.content_type != expected_mime:
            logger.warning(
                "file_upload_mime_mismatch",
                extension=extension,
                expected_mime=expected_mime,
                actual_mime=file.content_type,
                filename=filename,
            )
            raise HTTPException(
                status_code=415,
                detail=f"MIME type mismatch: extension .{extension} suggests {expected_mime}, "
                f"but received {file.content_type}.",
            )

    # ------------------------------------------------------------------
    # Validation successful – log at DEBUG level for traceability.
    # ------------------------------------------------------------------
    logger.debug(
        "upload_validation_passed",
        filename=file.filename,
        size_bytes=size,
        content_type=file.content_type,
    )
