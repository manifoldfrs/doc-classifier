"""src/core/config.py
###############################################################################
Application configuration
###############################################################################
This module provides a centralized configuration class using Pydantic for
parsing and validating settings from environment variables.
"""

from __future__ import annotations

import json

# stdlib
import os
from functools import lru_cache
from typing import Any, List, Optional, Set

# third-party
from pydantic import Field, field_validator, model_validator
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


# Helper for parsing comma-separated values
def _parse_csv_str(v: str) -> List[str]:
    """Parse a comma-separated string into a list of values."""
    return [x.strip() for x in v.split(",") if x.strip()]


# ---------------------------------------------------------------------------
# Environment *pre-processing* – normalise problematic variables before Pydantic
# ---------------------------------------------------------------------------

# Ensure ``ALLOWED_API_KEYS`` is **JSON** so that Pydantic can coerce it into a
# ``List[str]`` without raising ``SettingsError``.  Developers may prefer the
# more convenient comma-separated style in *.env* files which would otherwise
# trip Pydantic's strict JSON parser.

_env_api_keys = os.environ.get("ALLOWED_API_KEYS")
if _env_api_keys and "[" not in _env_api_keys:
    os.environ["ALLOWED_API_KEYS"] = json.dumps(_parse_csv_str(_env_api_keys))

# Normalise ``ALLOWED_EXTENSIONS`` for the same reason – maintain developer
# convenience while satisfying Pydantic's strict JSON requirement for complex
# types.

_env_allowed_ext = os.environ.get("ALLOWED_EXTENSIONS")
if _env_allowed_ext and "[" not in _env_allowed_ext:
    os.environ["ALLOWED_EXTENSIONS"] = json.dumps(_parse_csv_str(_env_allowed_ext))

# ---------------------------------------------------------------------------
# Settings model
# ---------------------------------------------------------------------------


class Settings(BaseSettings):
    """
    Application configuration settings, loaded from environment variables.
    """

    # Basic app settings
    debug: bool = False
    pipeline_version: str = "v0.1.0"
    commit_sha: Optional[str] = None
    prometheus_enabled: bool = True

    # API key configuration
    allowed_api_keys: List[str] = Field(default_factory=list)

    # File upload settings
    allowed_extensions_raw: str = (
        "pdf,docx,xlsx,xls,csv,jpg,jpeg,png,tiff,tif,gif,bmp,eml,msg,txt"
    )
    allowed_extensions: Set[str] = set()
    max_file_size_mb: int = 10
    max_batch_size: int = 50

    # Classification confidence settings
    confidence_threshold: float = 0.65
    early_exit_confidence: float = 0.9

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_nested_delimiter=None,
        json_schema_extra={"example": {"allowed_api_keys": ["key1", "key2"]}},
        # Disable automatic JSON parsing globally – custom validators will handle coercion.
        enable_decoding=False,
        protected_namespaces=("protect_", "private_"),
    )

    @model_validator(mode="after")
    def parse_settings(self) -> "Settings":
        """Process all settings after validation."""
        # Parse API keys from environment if needed
        if not self.allowed_api_keys and os.environ.get("ALLOWED_API_KEYS"):
            self.allowed_api_keys = _parse_csv_str(os.environ["ALLOWED_API_KEYS"])

        # Parse extensions
        if not self.allowed_extensions:
            if self.allowed_extensions_raw:
                self.allowed_extensions = {
                    ext.strip().lower().lstrip(".")
                    for ext in self.allowed_extensions_raw.split(",")
                    if ext.strip()
                }
            else:
                self.allowed_extensions = _DEFAULT_ALLOWED_EXTENSIONS

        return self

    @field_validator("early_exit_confidence")
    @classmethod
    def validate_confidence_thresholds(cls, v: float, info: Any) -> float:
        """Validate confidence thresholds (early_exit must be >= confidence)."""
        values = info.data
        confidence = values.get("confidence_threshold", 0.65)
        if v < confidence:
            raise ValueError("EARLY_EXIT_CONFIDENCE must be >= CONFIDENCE_THRESHOLD")
        return v

    def is_extension_allowed(self, extension: str) -> bool:
        """
        Check if file extension is allowed.

        Args:
            extension: The file extension to check (with or without leading dot)

        Returns:
            True if extension is allowed, False otherwise
        """
        if not extension:
            return False

        # Normalize extension (remove dot, lowercase)
        clean_ext = extension.lower().lstrip(".")
        return clean_ext in self.allowed_extensions

    @field_validator("allowed_api_keys", mode="before")
    @classmethod
    def _coerce_allowed_api_keys(cls, v: Any) -> List[str]:  # noqa: D401
        """Allow comma-separated string in addition to a proper JSON array."""

        if isinstance(v, str):
            return _parse_csv_str(v)
        if v is None:
            return []
        return v

    @field_validator("allowed_extensions", mode="before")
    @classmethod
    def _coerce_allowed_extensions(cls, v: Any) -> Set[str]:
        """Convert comma or JSON strings into a set[str]."""

        if v is None or v == "":
            return set()

        if isinstance(v, str):
            # Accept comma-separated or JSON string formats
            if v.strip().startswith("["):
                try:
                    parsed: list[str] = json.loads(v)
                    return {ext.strip().lower().lstrip(".") for ext in parsed if ext}
                except json.JSONDecodeError:
                    pass
            return {
                ext.strip().lower().lstrip(".") for ext in v.split(",") if ext.strip()
            }
        if isinstance(v, (list, set, tuple)):
            return {
                str(ext).strip().lower().lstrip(".") for ext in v if str(ext).strip()
            }
        return v


# ---------------------------------------------------------------------------
# Public accessor (lazy singleton)
# ---------------------------------------------------------------------------


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached instance of application settings.

    In test environments (when PYTEST_CURRENT_TEST is set),
    this bypasses the cache to provide fresh instances for each test.
    """
    # Always return a fresh instance when running pytest to avoid test side effects
    if "PYTEST_CURRENT_TEST" in os.environ:
        get_settings.cache_clear()
        settings = Settings()
        if os.environ.get("ALLOWED_API_KEYS") and not settings.allowed_api_keys:
            settings.allowed_api_keys = _parse_csv_str(os.environ["ALLOWED_API_KEYS"])
        return settings
    return Settings()
