from __future__ import annotations

import time
from typing import Awaitable, Callable, Dict

import structlog
from starlette.datastructures import UploadFile

from src.classification.stages import (
    stage_filename,
    stage_metadata,
    stage_ocr,
    stage_text,
)
from src.classification.types import ClassificationResult, StageOutcome
from src.core.config import get_settings

logger = structlog.get_logger(__name__)

STAGE_REGISTRY: list[Callable[[UploadFile], Awaitable["StageOutcome"]]] = [
    stage_filename,
    stage_metadata,
    stage_text,
    stage_ocr,
]


def _get_file_size(file: UploadFile) -> int:
    """
    Get the size of an uploaded file in bytes safely.

    Args:
        file: The uploaded file

    Returns:
        Size in bytes, or 0 if unable to determine.
    """
    try:
        current_pos = file.file.tell()
        file.file.seek(0, 2)  # Seek to end
        size = file.file.tell()
        file.file.seek(current_pos)  # Reset position
        return size
    except Exception as e:
        logger.error("get_file_size_error", filename=file.filename, error=str(e))
        return 0


async def _execute_stages(file: UploadFile) -> Dict[str, StageOutcome]:
    """
    Execute all registered classification stages sequentially on a file.

    Args:
        file: The uploaded file to classify

    Returns:
        Dictionary mapping stage function names to their StageOutcome.
    """
    results = {}
    for stage_func in STAGE_REGISTRY:
        stage_name = stage_func.__name__  # Use function name as key
        try:
            # Ensure file pointer is at the beginning for each stage
            await file.seek(0)
            outcome = await stage_func(file)
            results[stage_name] = outcome
            logger.debug(
                "stage_executed",
                stage=stage_name,
                filename=file.filename,
                outcome_label=outcome.label,
                outcome_confidence=outcome.confidence,
            )
        except Exception as e:
            logger.error(
                "stage_execution_error",
                stage=stage_name,
                filename=file.filename,
                error=str(e),
                exc_info=True,
            )
            # Record error but continue to next stage if possible
            results[stage_name] = StageOutcome(label=None, confidence=None)
    return results


async def classify(file: UploadFile) -> ClassificationResult:
    """
    Classify a document by running it through all registered classification stages.

    Args:
        file: The uploaded file to classify

    Returns:
        ClassificationResult containing the final label, confidence, and details.
    """
    start_time = time.perf_counter()
    settings = get_settings()

    filename = file.filename or "<unknown>"
    mime_type = file.content_type or "application/octet-stream"
    size_bytes = _get_file_size(file)

    # Execute all stages
    stage_outcomes = await _execute_stages(file)

    # Extract confidences for the final result payload
    stage_confidences = {
        stage_name: outcome.confidence for stage_name, outcome in stage_outcomes.items()
    }

    # Aggregate results from all stages
    # Import here to avoid potential early import issues if confidence depends on types
    from src.classification.confidence import aggregate_confidences

    label, confidence = aggregate_confidences(stage_outcomes, settings=settings)

    end_time = time.perf_counter()
    processing_ms = (end_time - start_time) * 1000

    # Create the final result object
    result = ClassificationResult(
        filename=filename,
        mime_type=mime_type,
        size_bytes=size_bytes,
        label=label,
        confidence=round(confidence, 4),  # Round confidence for consistency
        stage_confidences=stage_confidences,
        pipeline_version=settings.pipeline_version,
        processing_ms=round(processing_ms, 2),
        # warnings/errors could be populated by stages if needed later
    )

    # Log the final outcome
    logger.info(
        "classification_complete",
        filename=result.filename,
        label=result.label,
        confidence=result.confidence,
        processing_ms=result.processing_ms,
        pipeline_version=result.pipeline_version,
        # Log individual stage results for debugging
        stage_outcomes={
            name: (outcome.label, outcome.confidence)
            for name, outcome in stage_outcomes.items()
        },
    )

    return result
