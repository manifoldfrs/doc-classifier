"""src/api/routes/files.py
###############################################################################
/v1/files – File-upload & classification endpoint (Step 5.1)
###############################################################################
This router exposes a **single** POST endpoint that accepts one *or* many files
and returns an array of structured classification results.  It orchestrates the
entire ingestion → validation → classification workflow synchronously.  Async
batch off-loading (>10 files) will be implemented in Step 5.3; for now, the
route enforces the *MAX_BATCH_SIZE* limit declared in the environment.

Key behaviours
==============
1. **Upload validation** – every `UploadFile` passes through
   :pyfunc:`src.ingestion.validators.validate_file`.  Requests containing *any*
   invalid file abort immediately with the appropriate HTTP error.
2. **Concurrent classification** – files are processed concurrently via
   :pyfunc:`asyncio.gather` for maximal throughput while respecting FastAPI's
   async context.
3. **Request-level correlation** – the `X-Request-ID` header propagated by
   :class:`src.core.logging.RequestLoggingMiddleware` is injected into each
   result model so clients can cross-reference logs.
4. **Edge-case handling** –
   • Empty *files* field ⟶ HTTP 400.
   • Batch size > `MAX_BATCH_SIZE` ⟶ HTTP 413 (payload too large analogue).

The route deliberately **avoids** persistence, background tasks, or webhooks –
those concerns belong to later steps of the implementation plan.
"""

from __future__ import annotations

# stdlib
import asyncio
import uuid
from typing import List

# third-party
import structlog
from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse

# local
from src.api.schemas import ClassificationResultSchema
from src.classification import classify
from src.core.config import Settings, get_settings
from src.ingestion.validators import validate_file

__all__: list[str] = [
    "router",
]

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Router configuration
# ---------------------------------------------------------------------------
router = APIRouter(prefix="/v1", tags=["files"])


# ---------------------------------------------------------------------------
# Dependency helpers – keep business logic out of endpoint signature
# ---------------------------------------------------------------------------
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
    files: List[UploadFile] = File(  # noqa: B008
        ..., description="One or many files to classify"
    ),
    settings: Settings = Depends(get_settings),  # noqa: B008
) -> JSONResponse:
    """Validate **files**, run classification, respond with structured JSON.

    The route operates synchronously for ≤ 10 files (configurable in later
    steps).  Each file is validated *individually* before any classification
    begins – this fail-fast strategy prevents wasting CPU time on large batches
    when the request is going to be rejected anyway.
    """

    # ------------------------------------------------------------------
    # 1. Basic request-level validation
    # ------------------------------------------------------------------
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
    # 2. Per-file validation (extension, size, etc.)
    # ------------------------------------------------------------------
    for upload in files:
        validate_file(upload, settings=settings)

    # ------------------------------------------------------------------
    # 3. Run classification concurrently – gather preserves order
    # ------------------------------------------------------------------
    results: List[ClassificationResultSchema] = await asyncio.gather(
        *(_classify_single(f) for f in files)
    )

    # Inject the *same* request_id for all results (consistency across batch)
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
    for res in results:
        # `object.__setattr__` works because the model is frozen but mutable via
        # the private API – safe within server-side code.
        object.__setattr__(res, "request_id", request_id)

    logger.info(
        "batch_classification_complete",
        batch_size=len(files),
        request_id=request_id,
    )
    # FastAPI automatically serialises Pydantic models, but we return an explicit
    # JSONResponse so we can attach the request_id header.
    response = JSONResponse(content=[r.dict(by_alias=True) for r in results])
    response.headers["X-Request-ID"] = request_id
    return response
