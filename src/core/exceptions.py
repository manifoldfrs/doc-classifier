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
"""

from __future__ import annotations

__all__: list[str] = [
    "MetadataProcessingError",
    # ModelNotAvailableError is defined and exported from model.py
]


class MetadataProcessingError(Exception):
    """
    Raised when an error occurs during the metadata extraction or processing stage.

    This exception can be used to signal issues specifically within the
    `stage_metadata` part of the classification pipeline, differentiating
    them from more general file I/O errors or other pipeline stage failures.
    """

    pass
