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
