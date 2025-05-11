from __future__ import annotations

import uuid
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

__all__: list[str] = [
    "ClassificationResultSchema",
]

"""
Notes on field inheritance and representation:

We intentionally inherit from the *internal* model to avoid duplicating 10+
attribute definitions and to guarantee that any future changes to the internal
representation (e.g. adding `sha256`) will cause a **mypy** type error here
unless we explicitly update the public contract. This keeps the API and the
core pipeline in lock-step at compile-time.
"""


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

    filename: str
    mime_type: str
    size_bytes: int
    label: str
    confidence: float
    stage_confidences: Dict[str, Optional[float]] = Field(default_factory=dict)
    pipeline_version: str
    processing_ms: float

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
