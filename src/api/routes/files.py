"""src/api/routes/files.py
###############################################################################
/ v1/files – File-upload & classification endpoint (Step 5.1 + 5.3 async jobs)
###############################################################################
This router now supports **asynchronous batch processing** for large uploads as
part of Implementation Plan – *Step 5.3*.  When the number of files exceeds
``ASYNC_THRESHOLD`` the request is accepted with **202 Accepted** and a
``job_id`` that clients can poll via :http:get:`/v1/jobs/{job_id}`.
"""

from __future__ import annotations

# stdlib
import asyncio
import uuid
from typing import List, Tuple

# third-party
import structlog
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from fastapi.responses import JSONResponse

from src.api.routes.jobs import create_job, run_job  # async helpers

# local
from src.api.schemas import ClassificationResultSchema
from src.classification import classify
from src.core.config import Settings, get_settings
from src.ingestion.validators import validate_file
from src.utils.auth import verify_api_key  # NEW – auth dependency

__all__: list[str] = [
    "router",
]

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Router configuration – API-key authentication enforced globally
# ---------------------------------------------------------------------------
router = APIRouter(
    prefix="/v1",
    tags=["files"],
    dependencies=[Depends(verify_api_key)],
)

# Threshold beyond which uploads are processed asynchronously (demo-only)
ASYNC_THRESHOLD: int = 10


# ---------------------------------------------------------------------------
# Dependency helpers – keep business logic out of endpoint signature
# ---------------------------------------------------------------------------
async def _classify_single(
    file: UploadFile,
) -> ClassificationResultSchema:  # noqa: D401
    """Run full pipeline for **file** and wrap into public response schema."""

    internal_result = await classify(file)  # returns ClassificationResult
    return ClassificationResultSchema(**internal_result.dict())


# ---------------------------------------------------------------------------
# Internal helpers – keep public functions concise
# ---------------------------------------------------------------------------


async def _validate_batch(
    files: List[UploadFile], settings: Settings
) -> None:  # noqa: D401
    """Fail-fast validation (extension, size) for every file in *files*."""

    for upload in files:
        validate_file(upload, settings=settings)


async def _enqueue_async_job(
    files: List[UploadFile],
    background_tasks: BackgroundTasks,
    request: Request,
    settings: Settings,
) -> JSONResponse:  # noqa: D401
    """Queue **files** for background processing and return 202 response."""

    await _validate_batch(files, settings)

    snapshots: List[Tuple[str, str | None, bytes]] = []
    for upload in files:
        await upload.seek(0)
        snapshots.append(
            (
                upload.filename or "<unknown>",
                upload.content_type,
                await upload.read(),
            )
        )

    job_id: str = await create_job(len(files))
    background_tasks.add_task(run_job, job_id, snapshots)

    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content={
            "job_id": job_id,
            "status": "queued",
            "estimated_completion_seconds": max(2, len(files)),
        },
        headers={"X-Request-ID": request.headers.get("x-request-id", job_id)},
    )


async def _process_sync_batch(
    files: List[UploadFile],
    request: Request,
    settings: Settings,
) -> JSONResponse:  # noqa: D401
    """Process **files** immediately and return 200 response."""

    await _validate_batch(files, settings)

    results: List[ClassificationResultSchema] = await asyncio.gather(
        *(_classify_single(f) for f in files)
    )

    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
    for res in results:
        object.__setattr__(res, "request_id", request_id)

    logger.info(
        "batch_classification_complete",
        batch_size=len(files),
        request_id=request_id,
    )

    return JSONResponse(
        content=[r.dict(by_alias=True) for r in results],
        headers={"X-Request-ID": request_id},
    )


# ---------------------------------------------------------------------------
# Public endpoint
# ---------------------------------------------------------------------------
@router.post(
    "/files",
    summary="Classify one or many uploaded files.",
    response_model=List[ClassificationResultSchema],
    status_code=status.HTTP_200_OK,
)
async def upload_and_classify_files(  # noqa: D401 – FastAPI handler
    request: Request,
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(  # noqa: B008
        ..., description="One or many files to classify"
    ),
    settings: Settings = Depends(get_settings),  # noqa: B008
) -> JSONResponse:
    """Validate **files**, run classification, respond with structured JSON.

    Behaviour overview
    ------------------
    • **≤ ASYNC_THRESHOLD** files → processed synchronously (HTTP 200).
    • **> ASYNC_THRESHOLD** files → job queued (HTTP 202) and result retrievable
      via /v1/jobs/<job_id>.
    """

    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files supplied under form field 'files'.",
        )

    if len(files) > settings.max_batch_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"Batch size {len(files)} exceeds limit of "
                f"{settings.max_batch_size}."
            ),
        )

    # ------------------------------------------------------------------
    # Async pathway – large batch off-loaded to background task
    # ------------------------------------------------------------------
    if len(files) > ASYNC_THRESHOLD:
        return await _enqueue_async_job(files, background_tasks, request, settings)

    # ------------------------------------------------------------------
    # Synchronous pathway – small batches processed immediately
    # ------------------------------------------------------------------

    return await _process_sync_batch(files, request, settings)
