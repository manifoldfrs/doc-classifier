from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

__all__: list[str] = ["StageOutcome", "ClassificationResult"]


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
        warnings: List of non-fatal pipeline notices.
        errors: List of recoverable errors encountered.
    """

    filename: str
    mime_type: str
    size_bytes: int
    label: str
    confidence: float
    stage_confidences: Dict[str, Optional[float]] = field(default_factory=dict)
    pipeline_version: str = "v0.1.0"
    processing_ms: float = 0.0
    # Add warnings and errors to match schema/spec better internally
    warnings: List[Dict[str, str]] = field(default_factory=list)
    errors: List[Dict[str, str]] = field(default_factory=list)

    # Utility helpers
    def dict(self) -> dict[str, Any]:
        """Return a serialisable ``dict`` representation.

        The public API layer (``src.api.routes.*``) expects dataclass
        instances to expose a ``.dict()`` method similar to *Pydantic*
        models. Implementing the helper here avoids sprinkling
        ``dataclasses.asdict`` conversions throughout the code-base while
        keeping the domain model a plain dataclass.
        """
        from dataclasses import asdict

        # Exclude warnings/errors if empty, matching Pydantic's behavior
        # when dumping models with default empty lists.
        data = asdict(self)
        # Note: The public schema handles request_id, warnings, errors separately.
        # This internal dict is mostly for the API layer to convert to the schema.
        return data
