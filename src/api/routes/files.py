from __future__ import annotations

import asyncio
import uuid
from typing import List

import structlog
from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse

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


async def _classify_single(
    file: UploadFile,
) -> ClassificationResultSchema:  # noqa: D401
    """Run full pipeline for **file** and wrap into public response schema."""

    internal_result = await classify(file)  # returns ClassificationResult
    # The request_id will be patched by the endpoint once it knows the header.
    return ClassificationResultSchema(**internal_result.dict())


@router.post(
    "/files",
    summary="Classify one or many uploaded files.",
    response_model=List[ClassificationResultSchema],
    status_code=status.HTTP_200_OK,
)
async def upload_and_classify_files(  # noqa: D401 – FastAPI handler
    request: Request,
    files: List[UploadFile] = FILES_PARAM,
    settings: Settings = SETTINGS_DEP,
) -> JSONResponse:
    """Validate **files**, run classification, respond with structured JSON.

    The route operates synchronously for ≤ 10 files (configurable in later
    steps).  Each file is validated *individually* before any classification
    begins – this fail-fast strategy prevents wasting CPU time on large batches
    when the request is going to be rejected anyway.
    """

    if not files:  # pragma: no cover
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

    for upload in files:
        validate_file(upload, settings=settings)

    results: List[ClassificationResultSchema] = await asyncio.gather(
        *(_classify_single(f) for f in files)
    )

    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
    for res in results:
        # Ensure the request_id from the route invocation is set on each result item,
        # overriding the Pydantic model's default factory if it was used.
        # Pydantic V2: direct assignment works.
        res.request_id = request_id

    logger.info(
        "batch_classification_complete",
        batch_size=len(files),
        request_id=request_id,
    )

    # For Pydantic V2, use model_dump()
    response_payload = [r.model_dump(by_alias=True) for r in results]
    response = JSONResponse(content=response_payload)
    response.headers["X-Request-ID"] = request_id
    return response
