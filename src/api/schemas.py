"""src/api/schemas.py
###############################################################################
Public Pydantic models **exposed by the API layer**.
###############################################################################
This module defines the request/response schemas **visible to clients** of the
HeronAI document-classification micro-service.  Keeping them in a dedicated
file (rather than collocated with route handlers) ensures:

1. **Single source of truth** for OpenAPI generation and downstream
   integrations (e.g. Postman collections, typed SDKs).
2. **Re-usability** – the same model can be imported by multiple routers or
   background tasks without creating circular dependencies.
3. **Separation of concerns** – business logic lives in the route modules while
   IO-facing contracts live here.

Only a subset of the internal `ClassificationResult` produced by the pipeline
is surfaced directly.  Additional envelope fields (`request_id`, `warnings`,
`errors`) mandated by the technical specification are added here so that the
internal pipeline can evolve without breaking the public contract.
"""

from __future__ import annotations

# stdlib
import uuid
from typing import Dict, List, Optional

# third-party
from pydantic import BaseModel, Field

# local

__all__: list[str] = [
    "ClassificationResultSchema",
]


class _Warning(BaseModel):  # noqa: D101 – tiny data container
    code: str = Field(..., description="Machine-readable warning identifier.")
    message: str = Field(..., description="Human-readable explanation.")

    class Config:  # noqa: D106 – pydantic v1 config name
        frozen = True


class ClassificationResultSchema(BaseModel):
    """Public **Pydantic** representation of a single classification result.

    The schema mirrors :class:`src.classification.pipeline.ClassificationResult`
    while adding envelope fields mandated by the HTTP contract.
    """

    # Core result fields – duplicated here to stay independent from Pydantic
    filename: str
    mime_type: str
    size_bytes: int
    label: str
    confidence: float
    stage_confidences: Dict[str, Optional[float]] = Field(default_factory=dict)
    pipeline_version: str
    processing_ms: float

    # Envelope additions
    request_id: str = Field(
        default_factory=lambda: uuid.uuid4().hex,
        description="Request-scoped UUID used for log correlation.",
    )
    warnings: List[_Warning] = Field(
        default_factory=list,
        description="Non-fatal pipeline notices (e.g. OCR invoked).",
    )
    errors: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Recoverable errors encountered during processing.",
    )

    class Config:  # noqa: D106
        allow_population_by_field_name = True
        orm_mode = True


""" Notes on field inheritance and representation:
We intentionally inherit from the *internal* model to avoid duplicating 10+
attribute definitions and to guarantee that any future changes to the internal
representation (e.g. adding `sha256`) will cause a **mypy** type error here
unless we explicitly update the public contract. This keeps the API and the
core pipeline in lock-step at compile-time.
"""
