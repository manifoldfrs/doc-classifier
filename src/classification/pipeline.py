"""
Classification Pipeline Orchestrator

This module defines the main classification pipeline for the Document Classifier service.
It orchestrates the execution of various classification stages (filename,
metadata, text content, OCR) sequentially on an uploaded file. The pipeline
combines the outcomes from each stage using a confidence aggregation strategy
to produce a final classification label and score.

Key Responsibilities:
- Define the sequence of classification stages.
- Execute each stage on the input file.
- Aggregate stage outcomes into a final ClassificationResult.
- Handle errors during stage execution gracefully.
- Measure and report processing time.

Dependencies:
- `src.classification.stages`: Provides individual stage implementations.
- `src.classification.types`: Defines data structures for stage outcomes and final results.
- `src.classification.confidence`: Implements the logic for aggregating stage confidences.
- `src.core.config`: Provides application settings (thresholds, version).
- `structlog`: Used for structured logging throughout the pipeline execution.

"""

from __future__ import annotations

import time
from typing import Awaitable, Callable, Dict, List

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
from src.core.exceptions import StageExecutionError  # Domain-specific stage error

logger = structlog.get_logger(__name__)

# Define the sequence of stages to be executed in the pipeline.
# Stages are imported directly for clarity and explicit control over the pipeline flow.
STAGE_REGISTRY: List[Callable[[UploadFile], Awaitable["StageOutcome"]]] = [
    stage_filename,
    stage_metadata,
    stage_text,
    stage_ocr,
]


def _get_file_size(file: UploadFile) -> int:
    """
    Safely determine the size of an uploaded file in bytes.

    This helper function attempts to read the file size by seeking to the end.
    It handles potential exceptions and resets the file pointer.

    Args:
        file: The uploaded file object.

    Returns:
        The size of the file in bytes, or 0 if the size cannot be determined.
    """
    try:
        current_pos = file.file.tell()
        file.file.seek(0, 2)  # Seek to the end of the file
        size = file.file.tell()
        file.file.seek(current_pos)  # Reset file pointer to original position
        return size
    except (OSError, AttributeError, ValueError) as e:
        logger.error(
            "get_file_size_error", filename=file.filename, error=str(e), exc_info=True
        )
        return 0  # Return 0 if size check fails


async def _execute_stages(file: UploadFile) -> Dict[str, StageOutcome]:
    """
    Execute all registered classification stages sequentially on a file.

    Iterates through the `STAGE_REGISTRY`, calling each stage function.
    Ensures the file pointer is reset before each stage execution.
    Logs outcomes and errors for each stage.

    Args:
        file: The uploaded file to classify.

    Returns:
        A dictionary mapping stage function names to their respective `StageOutcome`.
        If a stage fails, its outcome will be recorded as (None, None).
    """
    results: Dict[str, StageOutcome] = {}
    for stage_func in STAGE_REGISTRY:
        stage_name = stage_func.__name__  # Use the function's name as the identifier
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
        except StageExecutionError as e:
            # Classification stage explicitly signalled a recoverable failure.
            # Log at *warning* rather than *error* to differentiate from
            # truly unexpected exceptions that will bubble up.
            logger.warning(
                "stage_execution_error",
                stage=stage_name,
                filename=file.filename,
                error=str(e),
            )
            # Record the failure but allow the pipeline to continue
            results[stage_name] = StageOutcome(label=None, confidence=None)
        except Exception as e:  # noqa: BLE001 – pipeline must isolate stage crashes
            # Any *unexpected* exception bubbling out of a stage is converted into a
            # generic StageExecutionError so the rest of the pipeline can proceed.
            logger.error(
                "stage_unexpected_exception",
                stage=stage_name,
                filename=file.filename,
                error=str(e),
                exc_info=True,
            )
            # Treat as non-recoverable stage failure but keep pipeline alive.
            results[stage_name] = StageOutcome(label=None, confidence=None)
    return results


async def classify(file: UploadFile) -> ClassificationResult:
    """
    Classify a document by running it through the defined pipeline stages.

    This is the main entry point for classifying a single uploaded file. It
    handles the entire process from executing stages to aggregating results
    and formatting the final output.

    Args:
        file: The uploaded file object (`starlette.datastructures.UploadFile`).

    Returns:
        A `ClassificationResult` object containing the final classification
        label, confidence score, stage-specific confidences, processing time,
        and other relevant metadata.
    """
    start_time = time.perf_counter()
    settings = get_settings()

    # Basic file metadata extraction
    filename = file.filename or "<unknown>"
    mime_type = file.content_type or "application/octet-stream"
    size_bytes = _get_file_size(file)

    # Execute all defined stages
    stage_outcomes = await _execute_stages(file)

    # Prepare stage confidences for the final result payload
    stage_confidences = {
        stage_name: outcome.confidence for stage_name, outcome in stage_outcomes.items()
    }

    # Aggregate results from all stages using the confidence module
    # Imported here to avoid potential circular dependency issues during startup
    from src.classification.confidence import aggregate_confidences

    label, confidence = aggregate_confidences(stage_outcomes, settings=settings)

    end_time = time.perf_counter()
    processing_ms = (end_time - start_time) * 1000

    # Construct the final result object
    result = ClassificationResult(
        filename=filename,
        mime_type=mime_type,
        size_bytes=size_bytes,
        label=label,
        confidence=round(confidence, 4),  # Round confidence for consistent output
        stage_confidences=stage_confidences,
        pipeline_version=settings.pipeline_version,
        processing_ms=round(processing_ms, 2),
        # Warnings and errors could potentially be populated by stages in future extensions
        warnings=[],
        errors=[],
    )

    # Log the final outcome of the classification process
    logger.info(
        "classification_complete",
        filename=result.filename,
        label=result.label,
        confidence=result.confidence,
        processing_ms=result.processing_ms,
        pipeline_version=result.pipeline_version,
        # Log individual stage outcomes (label, confidence) for detailed debugging
        stage_outcomes={
            name: (outcome.label, outcome.confidence)
            for name, outcome in stage_outcomes.items()
        },
    )

    return result
