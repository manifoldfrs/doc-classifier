from __future__ import annotations

# stdlib
import os
from functools import lru_cache
from typing import List, Set

# third-party
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

"""Application configuration module.

All environment variables required to run the *HeronAI Document Classifier* are
formalised in the :class:`Settings` model below.  The class is powered by
*Pydantic-Settings*, which provides a typed, declarative approach for
configuration management that is compatible across multiple environments
(local development, CI/CD, container runtimes, Vercel, etc.).

A **single source of truth** for configuration avoids the typical scattering of
`os.getenv` calls across the code-base and ensures:

1. **Discoverability** – every env var is documented in one location.
2. **Validation** – type coercion & validation happen once at start-time.
3. **Performance** – the model is instantiated once and cached via
   ``@lru_cache`` in ``src/core/__init__.py``.
4. **Testability** – tests can override env vars using standard monkeypatching.

The class is future-proofed for additional fields (database URLs, Redis, etc.)
without breaking existing deployments.
"""

# ---------------------------------------------------------------------------
# Defaults & constants
# ---------------------------------------------------------------------------

_DEFAULT_ALLOWED_EXTENSIONS: Set[str] = {
    "pdf",
    "docx",
    "doc",
    "xlsx",
    "xlsb",
    "xls",
    "csv",
    "jpg",
    "jpeg",
    "png",
    "txt",
    "md",
    "xml",
    "json",
    "html",
    "eml",
}


# ---------------------------------------------------------------------------
# Settings model
# ---------------------------------------------------------------------------


class Settings(BaseSettings):
    """Typed configuration read from environment variables.

    Field names are *snake_case* by convention.  Environment variables are
    specified in *UPPER_SNAKE_CASE* and linked to their respective fields via
    the ``alias`` parameter on :pyclass:`pydantic.Field`.
    """

    debug: bool = Field(
        False,
        alias="DEBUG",
        description="Enable verbose debugging & hot-reload features.",
    )
    # Raw env strings that are post-processed into structured properties.
    allowed_api_keys_raw: str = Field(
        "",
        alias="ALLOWED_API_KEYS",
        description="Comma-separated API keys (e.g. `key1,key2`).",
    )
    allowed_extensions_raw: str = Field(
        ",".join(sorted(_DEFAULT_ALLOWED_EXTENSIONS)),
        alias="ALLOWED_EXTENSIONS",
        description="Comma-separated list of lowercase extensions.",
    )

    # Upload / batch limits
    max_file_size_mb: int = Field(
        10,
        alias="MAX_FILE_SIZE_MB",
        description="Maximum size (in MB) of a single uploaded file.",
    )
    max_batch_size: int = Field(
        50,
        alias="MAX_BATCH_SIZE",
        description="Maximum number of files allowed in a single batch request.",
    )

    # Classification parameters
    confidence_threshold: float = Field(
        0.65,
        alias="CONFIDENCE_THRESHOLD",
        description="Minimum aggregated confidence score required to return a label.",
    )
    early_exit_confidence: float = Field(
        0.9,
        alias="EARLY_EXIT_CONFIDENCE",
        description="If any single stage exceeds this score the pipeline short-circuits.",
    )
    pipeline_version: str = Field(
        "v0.1.0",
        alias="PIPELINE_VERSION",
        description="Semantic version of the classification pipeline.",
    )

    # Observability
    prometheus_enabled: bool = Field(
        True,
        alias="PROMETHEUS_ENABLED",
        description="Toggle for Prometheus metric exposition.",
    )

    # Misc – automatically set by Vercel / CI, but defaults help local dev
    commit_sha: str | None = Field(
        None,
        alias="GIT_COMMIT_SHA",
        description="Short git SHA embedded into health/version endpoints.",
    )

    # Pydantic-Settings configuration
    model_config = SettingsConfigDict(
        case_sensitive=False,
        extra="ignore",  # Ignore unknown env vars to be forward compatible
        env_file=".env",  # Auto-load dotenv in local dev
    )

    # ------------------------------------------------------------------
    # Computed / derived properties – keep business-logic outside __init__.
    # ------------------------------------------------------------------

    @property
    def allowed_api_keys(self) -> List[str]:
        """Return the parsed list of allowed static API keys."""

        return [k.strip() for k in self.allowed_api_keys_raw.split(",") if k.strip()]

    @property
    def allowed_extensions(self) -> Set[str]:
        """Return the parsed set of lowercase file extensions."""

        return {
            e.strip().lower()
            for e in self.allowed_extensions_raw.split(",")
            if e.strip()
        }

    @model_validator(mode="after")
    def validate_confidence_thresholds(self) -> "Settings":
        """Ensure early_exit_confidence is not lower than confidence_threshold.

        This validator runs after all individual fields have been initialized
        and parsed. It checks the logical consistency between
        `early_exit_confidence` and `confidence_threshold`.
        """
        if self.early_exit_confidence < self.confidence_threshold:
            raise ValueError(
                "EARLY_EXIT_CONFIDENCE must be >= CONFIDENCE_THRESHOLD. "
                f"Got early_exit_confidence={self.early_exit_confidence}, "
                f"confidence_threshold={self.confidence_threshold}"
            )
        return self

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def is_extension_allowed(self, extension: str) -> bool:
        """Check whether a *file extension* is accepted by the service.

        Parameters
        ----------
        extension:
            The extension **without** leading dot, case insensitive.  E.g.
            ``"pdf"`` or ``"PDF"``.
        """

        return extension.lower() in self.allowed_extensions


# ---------------------------------------------------------------------------
# Public accessor (lazy singleton)
# ---------------------------------------------------------------------------


@lru_cache()
def get_settings() -> Settings:  # pragma: no cover – wrapper delegates to Settings()
    """Return a cached :class:`Settings` instance.

    Cache is cleared for tests if `PYTEST_CURRENT_TEST` is set to ensure
    test isolation with regards to settings.
    """
    # Check if running in a pytest environment and clear cache if so.
    # This is a common pattern to ensure tests get fresh settings if they modify environment variables.
    if "PYTEST_CURRENT_TEST" in os.environ:
        get_settings.cache_clear()
    return Settings()
