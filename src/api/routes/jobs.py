from __future__ import annotations

import asyncio
import uuid
from enum import Enum
from io import BytesIO
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.datastructures import UploadFile

from src.api.schemas import ClassificationResultSchema
from src.classification import classify
from src.utils.auth import verify_api_key

__all__: list[str] = [
    "router",
    "create_job",
    "run_job",
]


class JobStatus(str, Enum):  # noqa: D101 – simple enum wrapper
    queued = "queued"
    processing = "processing"
    done = "done"


class JobRecord:  # noqa: D101 – minimal container (no Pydantic to keep lightweight)
    def __init__(self, total_files: int):
        self.status: JobStatus = JobStatus.queued
        self.total_files: int = total_files
        self.results: List[ClassificationResultSchema] = []


# In-memory registry guarded by a single asyncio.Lock for atomic updates
_JOB_REGISTRY: Dict[str, JobRecord] = {}
_REGISTRY_LOCK: asyncio.Lock = asyncio.Lock()


after_complete_sleep: float = 0.01  # tiny yield to event loop between files


async def create_job(total_files: int) -> str:  # noqa: D401
    """Insert a *queued* job into the registry and return its **job_id**."""

    job_id: str = uuid.uuid4().hex
    async with _REGISTRY_LOCK:
        _JOB_REGISTRY[job_id] = JobRecord(total_files)
    return job_id


def _build_upload_from_bytes(
    filename: str,
    content_type: str | None,
    payload: bytes,
) -> UploadFile:  # noqa: D401 helper – tiny so lives here
    """Return an UploadFile wrapping **payload** so classifier can consume it."""

    return UploadFile(
        filename=filename,
        file=BytesIO(payload),
        content_type=content_type,
    )


async def run_job(
    job_id: str,
    raw_files: List[tuple[str, str | None, bytes]],
) -> None:  # noqa: D401 – background task entry-point
    """Process **raw_files** and populate the job record with results."""

    async with _REGISTRY_LOCK:
        record: Optional[JobRecord] = _JOB_REGISTRY.get(job_id)
        if record is None:
            # Job was deleted/cancelled – nothing to do
            return
        record.status = JobStatus.processing

    try:
        # Execute classification sequentially to avoid exhausting CPU.
        for filename, content_type, payload in raw_files:
            upload = _build_upload_from_bytes(filename, content_type, payload)
            try:
                internal_result = await classify(upload)
                record.results.append(
                    ClassificationResultSchema(**internal_result.dict())
                )
            except Exception as exc:  # noqa: BLE001 – capture per-file failures
                record.results.append(
                    ClassificationResultSchema(  # type: ignore[call-arg]
                        filename=filename,
                        mime_type=content_type,
                        size_bytes=len(payload),
                        label="error",
                        confidence=0.0,
                        stage_confidences={},
                        processing_ms=0.0,
                        pipeline_version="v1.0.0",
                        warnings=[],
                        errors=[{"code": "classification_error", "message": str(exc)}],
                    )
                )
            await asyncio.sleep(after_complete_sleep)
    finally:
        async with _REGISTRY_LOCK:
            record.status = JobStatus.done


router: APIRouter = APIRouter(
    prefix="/v1",
    tags=["jobs"],
    dependencies=[Depends(verify_api_key)],
)


@router.get(
    "/jobs/{job_id}",
    summary="Retrieve the status/results of an asynchronous batch job.",
)
async def get_job(job_id: str) -> JSONResponse:  # noqa: D401 – FastAPI handler
    """Return job **status** or final *results* when completed."""

    async with _REGISTRY_LOCK:
        record = _JOB_REGISTRY.get(job_id)

    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' not found.",
        )

    if record.status != JobStatus.done:
        return JSONResponse({"job_id": job_id, "status": record.status})

    return JSONResponse(
        {
            "job_id": job_id,
            "status": record.status,
            "results": [r.dict(by_alias=True) for r in record.results],
        }
    )
