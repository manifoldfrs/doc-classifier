from __future__ import annotations

import mimetypes
import os
from typing import Optional

import structlog
from fastapi import HTTPException, UploadFile

from src.core.config import Settings, get_settings

__all__: list[str] = ["validate_file"]

logger = structlog.get_logger(__name__)


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

    if not file.filename:
        logger.warning("file_upload_no_filename")
        raise HTTPException(status_code=400, detail="No filename provided.")

    filename = file.filename.lower()
    _, extension = os.path.splitext(filename)

    if not extension:
        logger.warning("file_upload_no_extension", filename=filename)
        raise HTTPException(status_code=415, detail="File has no extension.")

    extension = extension[1:]

    if not settings.is_extension_allowed(extension):
        logger.warning("file_upload_invalid_extension", extension=extension)
        raise HTTPException(
            status_code=415, detail=f"Unsupported file extension: .{extension}"
        )

    try:
        file.file.seek(0, 2)
        size = file.file.tell()
        file.file.seek(0)

        if size == 0:
            logger.warning("file_upload_empty", filename=filename)
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        max_size = settings.max_file_size_mb * 1024 * 1024
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

    if file.content_type and extension:
        expected_mime = mimetypes.guess_type(f"file.{extension}")[0]

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

    logger.debug(
        "upload_validation_passed",
        filename=file.filename,
        size_bytes=size,
        content_type=file.content_type,
    )
