"""
Core Custom Exceptions

This module defines domain-specific exceptions used throughout the HeronAI
application. Using custom exceptions allows for more granular error handling
and clearer communication of issues compared to relying solely on built-in
Python exceptions.

Key Benefits:
- Improved Readability: Custom exception names (e.g., `MetadataProcessingError`)
  immediately convey the context of the error.
- Granular Handling: Allows specific `try...except` blocks to catch particular
  application-level errors without catching overly broad `Exception` types.
- Consistent Error Reporting: Can be integrated with global error handlers
  in the API layer to provide consistent error responses to clients.

Defined Exceptions:
- `MetadataProcessingError`: Raised when an error occurs during the metadata
  extraction or processing stage of the classification pipeline.
- `ModelNotAvailableError`: (Already defined in `src.classification.model`)
  Raised when the persisted ML model artefact cannot be loaded.
- `StageExecutionError`: Raised by classification *stage* functions when an unrecoverable error occurs.
"""

from __future__ import annotations

__all__: list[str] = [
    "MetadataProcessingError",
    # ModelNotAvailableError is defined and exported from model.py
    "StageExecutionError",
]


class MetadataProcessingError(Exception):
    """
    Raised when an error occurs during the metadata extraction or processing stage.

    This exception can be used to signal issues specifically within the
    `stage_metadata` part of the classification pipeline, differentiating
    them from more general file I/O errors or other pipeline stage failures.
    """

    pass


class StageExecutionError(Exception):
    """Raised by classification *stage* functions when an unrecoverable error occurs.

    The error is intentionally broad at the *domain* level – it replaces ad-hoc
    ``except Exception`` catch-alls in the pipeline.  Individual stage modules
    should **only** raise this wrapper (or a narrower custom error) once they
    have applied any stage-specific recovery/fallback logic.
    """

    pass
