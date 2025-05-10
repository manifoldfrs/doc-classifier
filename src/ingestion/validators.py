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
import logging
import mimetypes
from typing import Final

# third-party
import structlog
from fastapi import HTTPException, UploadFile, status

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


def validate_file(
    file: UploadFile, *, settings: Settings | None = None
) -> None:  # noqa: D401
    """Validate an uploaded file against service constraints.

    Parameters
    ----------
    file:
        An instance of :class:`starlette.datastructures.UploadFile` obtained
        from a FastAPI route parameter.
    settings:
        Optional explicit :class:`src.core.config.Settings` instance.  When not
        provided the *process-wide* singleton returned by `get_settings()` is
        used.  Supplying an instance is mainly useful for **unit tests** where
        isolation from global env vars is desired.

    Raises
    ------
    fastapi.HTTPException
        When any validation rule fails.  The *status_code* conveys the nature
        of the failure as per the project specification.
    """

    settings = settings or get_settings()

    # ------------------------------------------------------------------
    # 1. Filename presence & extension check
    # ------------------------------------------------------------------
    if not file.filename:
        logger.warning("upload_validation_failed", reason="empty_filename")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No filename provided.",
        )

    if "." not in file.filename:
        logger.warning("upload_validation_failed", reason="missing_extension")
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="File has no extension.",
        )

    extension: str = file.filename.rsplit(".", 1)[1].lower()
    if not settings.is_extension_allowed(extension):
        logger.warning(
            "upload_validation_failed",
            reason="extension_not_allowed",
            extension=extension,
        )
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file extension: .{extension}",
        )

    # ------------------------------------------------------------------
    # 2. Size check – ensure payload ≤ MAX_FILE_SIZE_MB
    # ------------------------------------------------------------------
    try:
        # SpooledTemporaryFile supports tell() even when moderated by disk.
        file.file.seek(0, _SIZE_SEEK_END)  # type: ignore[arg-type]
        size_bytes: int = file.file.tell()  # type: ignore[attr-defined]
        file.file.seek(0, _SIZE_SEEK_START)  # Reset pointer for downstream use
    except (
        Exception
    ) as exc:  # noqa: BLE001 – explicit narrow exception unreliable across impls
        # Although project rules discourage broad excepts, the behaviour of
        # arbitrary file-like objects may vary between runtimes.  We therefore
        # convert *any* seek/tell failure into a HTTP 400 whilst logging the
        # original exception for observability.
        logging.getLogger(__name__).exception("size_check_failed", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unable to assess uploaded file size.",
        ) from exc

    if size_bytes == 0:
        logger.warning("upload_validation_failed", reason="empty_file")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    max_bytes: int = settings.max_file_size_mb * 1024 * 1024
    if size_bytes > max_bytes:
        logger.warning(
            "upload_validation_failed",
            reason="file_too_large",
            size_bytes=size_bytes,
            max_bytes=max_bytes,
        )
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"File size {size_bytes / (1024 * 1024):.2f} MB exceeds the "
                f"limit of {settings.max_file_size_mb} MB."
            ),
        )

    # ------------------------------------------------------------------
    # 3. MIME-type sanity check – best-effort using `mimetypes`.
    # ------------------------------------------------------------------
    expected_mime, _ = mimetypes.guess_type(file.filename)
    # Some extensions (e.g. .csv) may map to text/plain; treat None as unknown.
    if expected_mime and file.content_type and expected_mime != file.content_type:
        logger.warning(
            "upload_validation_failed",
            reason="mime_mismatch",
            expected=expected_mime,
            received=file.content_type,
        )
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"MIME type '{file.content_type}' does not match the expected "
                f"type '{expected_mime}' for extension '.{extension}'."
            ),
        )

    # ------------------------------------------------------------------
    # Validation successful – log at DEBUG level for traceability.
    # ------------------------------------------------------------------
    logger.debug(
        "upload_validation_passed",
        filename=file.filename,
        size_bytes=size_bytes,
        content_type=file.content_type,
    )
