from __future__ import annotations

# stdlib
from functools import lru_cache
from typing import Any, List, Set

# third-party
from dotenv import load_dotenv
from pydantic import Field, validator
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

load_dotenv(override=False)


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
    allowed_api_keys: List[str] = Field(
        default_factory=list,
        alias="ALLOWED_API_KEYS",
        description="Comma-delimited list of static API keys allowed to access the service.",
    )
    allowed_extensions: Set[str] = Field(
        default_factory=lambda: _DEFAULT_ALLOWED_EXTENSIONS.copy(),
        alias="ALLOWED_EXTENSIONS",
        description="Lower-case file extensions accepted by the upload endpoint (comma-separated).",
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
    )

    # ---------------------------------------------------------------------
    # Validators & post-init hooks
    # ---------------------------------------------------------------------

    @validator("allowed_api_keys", pre=True)
    def _split_api_keys(
        cls, v: str | List[str]
    ) -> List[str]:  # noqa: N805 – pydantic naming
        """Coerce a comma-separated string into a clean ``list[str]``.

        The transformation tolerates surrounding whitespace and empty strings.
        """

        if isinstance(v, list):
            return v
        return [key.strip() for key in v.split(",") if key.strip()]

    @validator("allowed_extensions", pre=True)
    def _split_extensions(cls, v: str | Set[str]) -> Set[str]:  # noqa: N805
        """Coerce comma-separated string into a ``set[str]`` of lowercase values."""

        if isinstance(v, set):
            return v
        return {ext.strip().lower() for ext in v.split(",") if ext.strip()}

    @validator("early_exit_confidence")
    def _early_exit_not_below_threshold(
        cls, v: float, values: dict[str, Any]
    ) -> float:  # noqa: N805
        """Ensure early-exit confidence is not lower than base threshold."""

        threshold = float(values.get("confidence_threshold", 0))
        if v < threshold:
            raise ValueError("EARLY_EXIT_CONFIDENCE must be >= CONFIDENCE_THRESHOLD")
        return v

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
    """Return a cached :class:`Settings` instance."""

    return Settings()
