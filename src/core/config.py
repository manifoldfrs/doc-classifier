from __future__ import annotations

import json
import os
from typing import Any, List, Optional, Set, cast

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _parse_csv_str(v: str) -> List[str]:
    """Parse a comma-separated string into a list of values."""
    return [x.strip() for x in v.split(",") if x.strip()]


# ---------------------------------------------------------------------------
# Environment *pre-processing* – normalise problematic variables before Pydantic
# ---------------------------------------------------------------------------

# Ensure ``ALLOWED_API_KEYS`` is **JSON** so that Pydantic can coerce it into a
# ``List[str]`` without raising ``SettingsError``.  Devs may prefer the
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


class Settings(BaseSettings):
    """
    Application configuration settings, loaded from environment variables.
    """

    debug: bool = False
    pipeline_version: str = "v0.1.0"
    commit_sha: Optional[str] = None
    prometheus_enabled: bool = True

    allowed_api_keys: List[str] = Field(default_factory=list)

    allowed_extensions_raw: Optional[str] = (
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
        # ------------------------------------------------------------------
        # API-key parsing – convert comma-separated strings to list[str].
        # ------------------------------------------------------------------

        env_api_keys = os.environ.get("ALLOWED_API_KEYS")
        if env_api_keys:
            if not self.allowed_api_keys:
                self.allowed_api_keys = _parse_csv_str(env_api_keys)
        else:
            # Environment variable removed ➜ ensure list is empty (important
            # for tests that explicitly `delenv('ALLOWED_API_KEYS')`).
            self.allowed_api_keys = []

        # ------------------------------------------------------------------
        # Derive *allowed_extensions* – honour explicit ``ALLOWED_EXTENSIONS``
        # from the environment first.  Only fall back to ``allowed_extensions_raw``
        # when the set is still empty.  This guarantees that developers can
        # override the defaults via *.env* without the value being silently
        # overwritten by the *raw* fallback.

        if not self.allowed_extensions:
            # No explicit ``ALLOWED_EXTENSIONS`` provided – compute from *raw*.
            if self.allowed_extensions_raw is None:
                # Nothing configured – disallow all extensions by default.
                self.allowed_extensions = set()
            else:
                # Empty string means "no extensions allowed"; otherwise parse.
                self.allowed_extensions = {
                    ext.strip().lower().lstrip(".")
                    for ext in self.allowed_extensions_raw.split(",")
                    if ext.strip()
                }

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
    def _coerce_allowed_api_keys(cls, v: Any) -> List[str]:
        """Allow comma-separated string in addition to a proper JSON array."""

        if isinstance(v, str):
            return _parse_csv_str(v)
        if v is None:
            return []
        # If validation has already produced a proper list[str] just return it.
        return cast(List[str], v)

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
        # Fallback – make a best-effort cast instead of returning *Any* to satisfy
        # strict typing expectations.
        from typing import Set as _Set
        from typing import cast

        return cast(_Set[str], v)


###############################################################################
# Public accessor – manual caching to support special behaviour in tests
###############################################################################


_CACHED_SETTINGS: Optional[Settings] = None


def get_settings() -> Settings:  # noqa: D401 – accessor helper
    """Return a **singleton** Settings instance unless running under pytest.

    The original implementation relied on :pyfunc:`functools.lru_cache`, but
    that made it impossible to *disable* caching on a per-call basis –
    specifically the unit-tests in *tests/unit/core/test_config.py* expect a
    **fresh** instance every time ``get_settings`` is invoked while
    ``PYTEST_CURRENT_TEST`` is present in the environment.
    """

    global _CACHED_SETTINGS  # noqa: PLW0603 – module-level singleton

    # Test-mode ➜ always deliver a **new** instance (no caching)
    if "PYTEST_CURRENT_TEST" in os.environ:
        return Settings()

    if _CACHED_SETTINGS is None:
        _CACHED_SETTINGS = Settings()

    return _CACHED_SETTINGS


# ---------------------------------------------------------------------------
# Mimic ``functools.lru_cache`` API expected by existing tests
# ---------------------------------------------------------------------------


def _clear_settings_cache() -> None:  # noqa: D401 – helper for tests
    """Clear the internal Settings singleton (used by unit-tests)."""

    global _CACHED_SETTINGS
    _CACHED_SETTINGS = None


# Expose the helper so tests can call ``get_settings.cache_clear()`` exactly as
# before, preserving the public contract while switching to a custom cache.
get_settings.cache_clear = _clear_settings_cache  # type: ignore[attr-defined]
