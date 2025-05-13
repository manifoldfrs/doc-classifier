from __future__ import annotations

import asyncio
import uuid
from typing import TYPE_CHECKING, Any, Dict, List

import redis.asyncio as aioredis
import structlog
from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse

from src.api.routes.jobs import create_job as create_redis_job
from src.api.routes.jobs import get_redis_client  # Import the Redis client dependency
from src.api.routes.jobs import run_job as run_redis_job
from src.api.schemas import ClassificationResultSchema
from src.classification import classify
from src.core.config import Settings, get_settings
from src.ingestion.validators import validate_file

__all__: list[str] = [
    "router",
]

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/v1", tags=["files"])

FILES_PARAM: List[UploadFile] = File(..., description="One or many files to classify")

SETTINGS_DEP: Settings = Depends(get_settings)
if TYPE_CHECKING:
    RedisT = aioredis.Redis[Any]
else:
    RedisT = aioredis.Redis  # type: ignore[misc]
REDIS_DEP: RedisT = Depends(get_redis_client)  # Added Redis client dependency


async def _classify_single(
    file: UploadFile, request_id: str  # Pass request_id for consistent logging/tracing
) -> ClassificationResultSchema:
    """Run full pipeline for **file** and wrap into public response schema."""
    internal_result = await classify(file)  # returns ClassificationResult
    # The request_id is now explicitly set.
    return ClassificationResultSchema(**internal_result.dict(), request_id=request_id)


@router.post(
    "/files",
    summary="Classify one or many uploaded files.",
    # Response model is dynamic: List[ClassificationResultSchema] for sync, or Job ID for async
    # Set to None here and handle response manually for flexibility.
    response_model=None,
    status_code=status.HTTP_200_OK,  # Default, will be overridden for async
)
async def upload_and_classify_files(
    request: Request,
    files: List[UploadFile] = FILES_PARAM,
    settings: Settings = SETTINGS_DEP,
    redis_client: RedisT = REDIS_DEP,  # Inject Redis client
) -> JSONResponse:
    """
    Validate files, then run classification synchronously for small batches
    or asynchronously via Redis-backed job queue for larger batches.
    """
    request_id_header = request.headers.get("x-request-id")
    # Generate a new UUID if x-request-id is not provided in the header
    # This ensures every batch operation has a unique ID.
    # For sync operations, individual file results will share this batch request_id.
    # For async, this becomes the basis for the job_id or part of its context.
    batch_request_id = request_id_header or uuid.uuid4().hex

    if not files:
        # This case should ideally be caught by FastAPI's validation of `List[UploadFile] = File(...)`
        # which requires at least one file. If it somehow passes, this is a fallback.
        logger.warning("upload_no_files_supplied", request_id=batch_request_id)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files supplied under form field 'files'.",
        )

    if len(files) > settings.max_batch_size:
        logger.warning(
            "upload_batch_too_large",
            request_id=batch_request_id,
            num_files=len(files),
            max_files=settings.max_batch_size,
        )
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"Batch size {len(files)} exceeds limit of "
                f"{settings.max_batch_size}."
            ),
        )

    # Validate all files upfront
    for upload in files:
        try:
            validate_file(upload, settings=settings)
        except HTTPException as e:
            logger.warning(
                "upload_validation_failed_http",
                request_id=batch_request_id,
                filename=upload.filename,
                detail=e.detail,
            )
            # Return a validation error response with the batch request ID
            error_response = JSONResponse(
                status_code=e.status_code,
                content={
                    "error": {
                        "code": e.status_code,
                        "message": e.detail,
                        "request_id": batch_request_id,
                    },
                    "detail": e.detail,
                },
            )
            error_response.headers["X-Request-ID"] = batch_request_id
            return error_response

    # Asynchronous processing for batches larger than 10 files
    # Note: project spec "Batch Processing" section User Story says "up to 50 files",
    # "Implementation" says "optionally spawn background task if >10 files".
    # Using 10 as threshold for async as per implementation detail.
    ASYNC_THRESHOLD = 10
    if len(files) > ASYNC_THRESHOLD:
        logger.info(
            "upload_async_batch_initiated",
            request_id=batch_request_id,
            num_files=len(files),
        )
        # Collect file data for the background task
        # We must read file contents here as UploadFile objects might not be safe
        # to pass directly to a background task that runs later.
        raw_files_data: List[tuple[str, str | None, bytes]] = []
        for f in files:
            await f.seek(0)  # Ensure pointer is at the start
            content_bytes = await f.read()
            # Ensure filename is a string (Starlette guarantees this)
            name: str = f.filename or "unknown"
            raw_files_data.append((name, f.content_type, content_bytes))
            await f.close()  # Close file after reading its content

        try:
            job_id = await create_redis_job(len(raw_files_data), redis_client)
            # Ensure the task uses the correct settings instance
            # Background task will use the same settings as the main app context
            asyncio.create_task(
                run_redis_job(job_id, raw_files_data, redis_client, settings)
            )
            async_response_payload: Dict[str, str] = {
                "job_id": job_id,
                "status": "queued",
            }
            response = JSONResponse(
                content=async_response_payload, status_code=status.HTTP_202_ACCEPTED
            )
            response.headers["X-Request-ID"] = (
                batch_request_id  # Use the batch_request_id
            )
            return response
        except (
            HTTPException
        ) as e:  # Catch Redis connection issues from create_redis_job
            # Re-raise to ensure proper response with X-Request-ID
            json_response = JSONResponse(
                status_code=e.status_code,
                content={
                    "error": {
                        "code": e.status_code,
                        "message": e.detail,
                        "request_id": batch_request_id,
                    },
                    "detail": e.detail,
                },
            )
            json_response.headers["X-Request-ID"] = batch_request_id
            return json_response

    # Synchronous processing for smaller batches
    logger.info(
        "upload_sync_batch_processing",
        request_id=batch_request_id,
        num_files=len(files),
    )
    # For synchronous, each file processing is part of the same main request.
    # The `batch_request_id` applies to the whole batch.
    # Each ClassificationResultSchema will have this ID.
    results: List[ClassificationResultSchema] = await asyncio.gather(
        *(_classify_single(f, batch_request_id) for f in files)
    )
    # Files are automatically closed by FastAPI after the request.

    logger.info(
        "batch_classification_complete_sync",
        batch_size=len(files),
        request_id=batch_request_id,
    )

    response_payload = [r.model_dump(by_alias=True) for r in results]
    response = JSONResponse(content=response_payload, status_code=status.HTTP_200_OK)
    response.headers["X-Request-ID"] = batch_request_id
    return response
