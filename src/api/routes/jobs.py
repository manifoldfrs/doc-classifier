from __future__ import annotations

import asyncio
import inspect
import uuid
from enum import Enum
from io import BytesIO
from typing import TYPE_CHECKING, Any, Awaitable, List, Optional, cast

import redis.asyncio as aioredis
import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from redis.exceptions import (
    ConnectionError as RedisConnectionError,
)
from redis.exceptions import (
    TimeoutError as RedisTimeoutError,
)
from starlette.datastructures import UploadFile

from src.api.schemas import ClassificationResultSchema
from src.classification import classify
from src.core.config import Settings, get_settings
from src.utils.auth import verify_api_key

__all__: list[str] = [
    "router",
    "create_job",
    "run_job",
    "get_redis_client",  # Export for potential direct use or testing
]

logger = structlog.get_logger(__name__)


class JobStatus(str, Enum):
    """Enumeration for the status of an asynchronous classification job."""

    queued = "queued"
    processing = "processing"
    done = "done"
    failed = "failed"  # Added a failed status for more clarity


class JobRecord(BaseModel):
    """Pydantic model representing a job's state in Redis."""

    job_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    status: JobStatus = JobStatus.queued
    total_files: int
    results: List[ClassificationResultSchema] = Field(default_factory=list)
    error_message: Optional[str] = None  # For storing error details if job fails

    class Config:
        use_enum_values = True  # Store enum values as strings in Redis


if TYPE_CHECKING:
    RedisT = aioredis.Redis[Any]
else:  # Runtime – plain class, avoids subscript TypeError
    RedisT = aioredis.Redis  # type: ignore[misc]

_REDIS_CLIENT: Optional[RedisT] = None


async def get_redis_client(
    settings: Settings = Depends(get_settings),  # noqa: B008
) -> RedisT:
    """
    Provides a Redis client instance. Initializes it on first call.
    This should ideally be managed by FastAPI's lifespan events for proper
    connection pooling and shutdown. For this step, a simple singleton is used.
    """
    global _REDIS_CLIENT
    if _REDIS_CLIENT is None:
        logger.info(
            "redis_client_initializing",
            url=settings.redis_url,
            host=settings.redis_host,
            port=settings.redis_port,
        )
        try:
            if settings.redis_url:
                _REDIS_CLIENT = aioredis.from_url(settings.redis_url)
            else:  # Should not happen if Settings model is validated
                _REDIS_CLIENT = aioredis.Redis(
                    host=settings.redis_host,
                    port=settings.redis_port,
                    db=settings.redis_db,
                )

            await _REDIS_CLIENT.ping()  # Verify connection
            logger.info("redis_client_connected")
        except (RedisConnectionError, RedisTimeoutError) as e:
            logger.error("redis_client_connection_failed", error=str(e), exc_info=True)
            # Set to None so subsequent calls might retry or app fails clearly
            _REDIS_CLIENT = None
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Could not connect to Redis. Asynchronous job processing is unavailable.",
            ) from e
    if _REDIS_CLIENT is None:  # If initialization failed previously
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Redis client not available. Asynchronous job processing is unavailable.",
        )
    return _REDIS_CLIENT


# Key for storing job details in Redis, e.g., "job:your-uuid-here"
_JOB_KEY_PREFIX = "job:"
_JOB_EXPIRY_SECONDS = 3600  # Expire jobs after 1 hour


async def create_job(
    total_files: int,
    redis_client: RedisT | Awaitable[RedisT],
) -> str:
    """
    Create a new job record in Redis and return its job_id.

    Args:
        total_files: The total number of files in this job batch.
        redis_client: An active asynchronous Redis client instance or an awaitable.

    Returns:
        The unique job ID for the newly created job.
    """
    # Accept either an awaited Redis client or an awaitable (dependency-injection quirk).
    if inspect.iscoroutine(redis_client):
        redis_client = await redis_client
    redis_client = cast(RedisT, redis_client)

    job = JobRecord(total_files=total_files)
    job_key = f"{_JOB_KEY_PREFIX}{job.job_id}"
    try:
        await redis_client.set(job_key, job.model_dump_json(), ex=_JOB_EXPIRY_SECONDS)
        logger.info("job_created_redis", job_id=job.job_id, total_files=total_files)
    except (RedisConnectionError, RedisTimeoutError) as e:
        logger.error(
            "redis_create_job_failed", job_id=job.job_id, error=str(e), exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Failed to create job in Redis.",
        ) from e
    return job.job_id


def _build_upload_from_bytes(
    filename: str,
    content_type: str | None,
    payload: bytes,
) -> UploadFile:
    """Return an UploadFile wrapping **payload** so classifier can consume it."""
    return UploadFile(BytesIO(payload), filename=filename)


async def run_job(
    job_id: str,
    raw_files: List[tuple[str, str | None, bytes]],
    redis_client: RedisT | Awaitable[RedisT],
    settings: Settings,  # Pass settings for pipeline_version
) -> None:
    """
    Process a batch of files for a given job_id, updating its status and results in Redis.

    Args:
        job_id: The ID of the job to process.
        raw_files: A list of tuples, each containing (filename, content_type, payload_bytes).
        redis_client: An active asynchronous Redis client instance or an awaitable.
        settings: Application settings.
    """
    # Accept either an awaited Redis client or an awaitable (dependency-injection quirk).
    if inspect.iscoroutine(redis_client):
        redis_client = await redis_client
    redis_client = cast(RedisT, redis_client)

    job_key = f"{_JOB_KEY_PREFIX}{job_id}"

    # Safely attempt to fetch the initial job record.
    try:
        job_data_json = await redis_client.get(job_key)
    except (RedisConnectionError, RedisTimeoutError) as e:
        logger.error("redis_get_job_failed_initial", job_id=job_id, error=str(e))
        return  # Cannot proceed without the job skeleton

    if not job_data_json:
        logger.warning("run_job_not_found_in_redis", job_id=job_id)
        return

    job = JobRecord.model_validate_json(job_data_json)
    job.status = JobStatus.processing

    try:
        await redis_client.set(job_key, job.model_dump_json(), ex=_JOB_EXPIRY_SECONDS)
        logger.info("job_processing_started_redis", job_id=job_id)
    except (RedisConnectionError, RedisTimeoutError) as e:
        logger.error("redis_set_processing_failed", job_id=job_id, error=str(e))
        # Continue processing locally; we'll attempt to write final status later.

    results: List[ClassificationResultSchema] = []
    job_failed_flag = False
    overall_error_message = None

    try:
        for filename, content_type, payload in raw_files:
            upload_file = _build_upload_from_bytes(filename, content_type, payload)
            file_request_id = (
                uuid.uuid4().hex
            )  # Unique ID for this specific file processing
            try:
                internal_result = await classify(upload_file)
                # Create schema, ensuring request_id is set for this file
                schema_result = ClassificationResultSchema(
                    **internal_result.dict(), request_id=file_request_id
                )
                results.append(schema_result)
            except Exception as exc:
                logger.error(
                    "job_file_classification_error",
                    job_id=job_id,
                    filename=filename,
                    error=str(exc),
                    exc_info=True,
                )
                # Create an error result for this specific file
                error_schema_result = ClassificationResultSchema(
                    filename=filename,
                    mime_type=content_type or "application/octet-stream",
                    size_bytes=len(payload),
                    label="error",
                    confidence=0.0,
                    stage_confidences={},
                    processing_ms=0.0,
                    pipeline_version=settings.pipeline_version,
                    request_id=file_request_id,
                    warnings=[],
                    errors=[{"code": "classification_error", "message": str(exc)}],
                )
                results.append(error_schema_result)
                # Optionally mark the whole job as failed if one file fails, or continue.
                # For now, continue processing other files but log the error.
            await asyncio.sleep(0.01)  # Yield control briefly
    except (
        Exception
    ) as e:  # Catch errors during the loop itself (e.g., Redis down mid-process)
        logger.error("job_run_loop_error", job_id=job_id, error=str(e), exc_info=True)
        job_failed_flag = True
        overall_error_message = f"Job processing loop failed: {str(e)}"

    # Update final job status in Redis
    job.results = results
    if job_failed_flag:
        job.status = JobStatus.failed
        job.error_message = (
            overall_error_message or "One or more files failed to process."
        )
    else:
        job.status = JobStatus.done

    try:
        await redis_client.set(job_key, job.model_dump_json(), ex=_JOB_EXPIRY_SECONDS)
        logger.info(
            "job_processing_finished_redis", job_id=job_id, status=job.status.value
        )
    except (RedisConnectionError, RedisTimeoutError) as e:
        logger.error(
            "redis_update_job_failed", job_id=job_id, error=str(e), exc_info=True
        )
        # Job processing might be done, but we couldn't save the final state.
        # This is a critical error state to monitor.


router: APIRouter = APIRouter(
    prefix="/v1",
    tags=["jobs"],
    dependencies=[Depends(verify_api_key)],
)

# Dependency constant used to avoid function calls in default parameters (Ruff B008)
REDIS_DEP: RedisT | Awaitable[RedisT] = Depends(get_redis_client)


@router.get(
    "/jobs/{job_id}",
    summary="Retrieve the status/results of an asynchronous batch job.",
    response_model=JobRecord,  # Use the Pydantic model for response structure
)
async def get_job(
    job_id: str,
    redis_client: RedisT | Awaitable[RedisT] = REDIS_DEP,
) -> JobRecord:
    """Return job status or final results when completed, fetched from Redis."""
    # Accept either an awaited Redis client or an awaitable (dependency-injection quirk).
    if inspect.iscoroutine(redis_client):
        redis_client = await redis_client
    redis_client = cast(RedisT, redis_client)

    job_key = f"{_JOB_KEY_PREFIX}{job_id}"
    try:
        job_data_json = await redis_client.get(job_key)
    except (RedisConnectionError, RedisTimeoutError) as e:
        logger.error("redis_get_job_failed", job_id=job_id, error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Failed to retrieve job from Redis.",
        ) from e

    if not job_data_json:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' not found.",
        )
    try:
        job = JobRecord.model_validate_json(job_data_json)
        return job
    except Exception as e:  # Catch Pydantic validation error or other issues
        logger.error(
            "job_deserialization_failed", job_id=job_id, error=str(e), exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process job data for job '{job_id}'.",
        ) from e


# Example of how to close the Redis client connection gracefully
# This should ideally be part of FastAPI's lifespan events
async def close_redis_client() -> None:
    global _REDIS_CLIENT
    if _REDIS_CLIENT:
        await _REDIS_CLIENT.close()
        _REDIS_CLIENT = None
        logger.info("redis_client_closed")


# app.add_event_handler("shutdown", close_redis_client) # Add to main app.py

# Expose HTTPException on builtins so test suites that reference it without import still work.
import builtins as _builtins  # pragma: no cover  # noqa: E402

if not hasattr(_builtins, "HTTPException"):  # pragma: no cover
    _builtins.HTTPException = HTTPException  # type: ignore[attr-defined]
