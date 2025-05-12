from __future__ import annotations

import mimetypes
import os
from typing import Optional

import structlog
from fastapi import HTTPException, UploadFile, status

from src.core.config import Settings, get_settings

__all__: list[str] = ["validate_file"]

logger = structlog.get_logger(__name__)


def _validate_filename(filename: Optional[str]) -> str:
    """Ensure filename exists and is not empty."""
    if not filename:
        logger.warning("file_upload_no_filename")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="No filename provided."
        )
    return filename.lower()


def _validate_extension(filename: str, settings: Settings) -> str:
    """Validate the file extension."""
    _, extension = os.path.splitext(filename)
    if not extension:
        logger.warning("file_upload_no_extension", filename=filename)
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="File has no extension.",
        )

    extension = extension[1:]  # Remove the leading dot
    if not settings.is_extension_allowed(extension):
        logger.warning(
            "file_upload_invalid_extension", extension=extension, filename=filename
        )
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file extension: .{extension}",
        )
    return extension


def _validate_size(file: UploadFile, filename: str, settings: Settings) -> int:
    """Validate the file size."""
    try:
        # Use file.seek(0, 2) and file.tell() to get size efficiently
        # This mimics the typical way size is checked for UploadFile
        current_pos = file.file.tell()
        file.file.seek(0, 2)
        size = file.file.tell()
        file.file.seek(current_pos)  # Reset position

        if size == 0:
            logger.warning("file_upload_empty", filename=filename)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty.",
            )

        max_size = settings.max_file_size_mb * 1024 * 1024
        if size > max_size:
            logger.warning(
                "file_upload_too_large",
                size=size,
                max_size=max_size,
                filename=filename,
            )
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=(
                    f"File size {size / 1024 / 1024:.2f} MB exceeds the "
                    f"limit of {settings.max_file_size_mb} MB."
                ),
            )
        return size
    except OSError as e:
        logger.error("file_size_check_failed", error=str(e), filename=filename)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unable to assess uploaded file size. The upload may be corrupted.",
        ) from e


def _validate_mime(
    file: UploadFile, extension: str, filename: str, settings: Settings
) -> None:
    """Validate the file's MIME type against its extension."""
    if file.content_type and extension:
        # Use mimetypes to guess the expected MIME type based on the validated extension
        expected_mime = mimetypes.guess_type(f"file.{extension}")[0]

        # Only raise an error if we have an expected MIME type and it doesn't match
        if expected_mime and file.content_type != expected_mime:
            logger.warning(
                "file_upload_mime_mismatch",
                extension=extension,
                expected_mime=expected_mime,
                actual_mime=file.content_type,
                filename=filename,
            )
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=(
                    f"MIME type mismatch: extension .{extension} suggests {expected_mime}, "
                    f"but received {file.content_type}."
                ),
            )


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
            - 400: Empty file or missing filename, size check error
            - 413: File too large
            - 415: Unsupported extension or MIME type mismatch, no extension
    """
    settings = settings or get_settings()

    # Chain validation steps
    filename_lower = _validate_filename(file.filename)
    extension = _validate_extension(filename_lower, settings)
    size = _validate_size(file, filename_lower, settings)
    _validate_mime(file, extension, filename_lower, settings)

    # Log success only after all validations pass
    logger.debug(
        "upload_validation_passed",
        filename=file.filename,  # Log original filename for clarity
        size_bytes=size,
        content_type=file.content_type,
    )
