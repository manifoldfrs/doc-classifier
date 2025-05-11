"""src/classification/pipeline.py
###############################################################################
Classification pipeline orchestrator
###############################################################################
This module contains the main classification pipeline implementation,
orchestrating multiple classification stages and aggregating their results.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import structlog
from starlette.datastructures import UploadFile

from src.core.config import get_settings

# Initialize logger
logger = structlog.get_logger(__name__)

# Registry for classification stages
STAGE_REGISTRY: List[Callable] = []


@dataclass
class StageOutcome:
    """
    Result from a single classification stage.

    Attributes:
        label: The document type label identified by the stage, or None
        confidence: Confidence score between 0.0-1.0, or None if no match
    """

    label: Optional[str] = None
    confidence: Optional[float] = None


@dataclass
class ClassificationResult:
    """
    Complete document classification result.

    Attributes:
        filename: Original filename of the document
        mime_type: MIME type of the document
        size_bytes: File size in bytes
        label: Final document type classification
        confidence: Final confidence score (0.0-1.0)
        stage_confidences: Confidence scores from each stage
        pipeline_version: Version of the classification pipeline
        processing_ms: Time taken to classify in milliseconds
    """

    filename: str
    mime_type: str
    size_bytes: int
    label: str
    confidence: float
    stage_confidences: Dict[str, Optional[float]] = field(default_factory=dict)
    pipeline_version: str = "v0.1.0"
    processing_ms: float = 0.0


def _get_file_size(file: UploadFile) -> int:
    """
    Get the size of an uploaded file in bytes.

    Args:
        file: The uploaded file

    Returns:
        Size in bytes
    """
    # Get current position
    current_pos = file.file.tell()

    # Seek to end
    file.file.seek(0, 2)

    # Get size
    size = file.file.tell()

    # Reset position
    file.file.seek(current_pos)

    return size


async def _execute_stages(file: UploadFile) -> Dict[str, StageOutcome]:
    """
    Execute all registered classification stages on a file.

    Args:
        file: The uploaded file to classify

    Returns:
        Dictionary mapping stage names to their outcomes
    """
    results = {}

    for stage in STAGE_REGISTRY:
        stage_name = stage.__name__
        outcome = await stage(file)
        results[stage_name] = outcome

    return results


async def classify(file: UploadFile) -> ClassificationResult:
    """
    Classify a document by running it through all registered classification stages.

    Args:
        file: The uploaded file to classify

    Returns:
        ClassificationResult with document type and confidence score
    """
    start_time = time.perf_counter()
    settings = get_settings()

    # Get file metadata
    size_bytes = _get_file_size(file)
    filename = file.filename or "<unknown>"
    mime_type = file.content_type or "application/octet-stream"

    # Execute all registered stages
    stage_outcomes = await _execute_stages(file)

    # Extract stage confidences for result
    stage_confidences = {
        stage_name: outcome.confidence for stage_name, outcome in stage_outcomes.items()
    }

    # Import here to break circular import
    from src.classification.confidence import aggregate_confidences

    # Aggregate confidences to determine final label and confidence
    label, confidence = aggregate_confidences(stage_outcomes, settings=settings)

    # Calculate processing time
    end_time = time.perf_counter()
    processing_ms = (end_time - start_time) * 1000

    # Create and return result
    result = ClassificationResult(
        filename=filename,
        mime_type=mime_type,
        size_bytes=size_bytes,
        label=label,
        confidence=confidence,
        stage_confidences=stage_confidences,
        pipeline_version=settings.pipeline_version,
        processing_ms=processing_ms,
    )

    # Log classification result
    logger.info(
        "classification_complete",
        filename=filename,
        label=label,
        confidence=confidence,
        processing_ms=processing_ms,
        pipeline_version=settings.pipeline_version,
    )

    return result
