# ruff: noqa: E402
from __future__ import annotations

import sys
from pathlib import Path

# Ensure repository root is first on sys.path
_repo_root: Path = Path(__file__).resolve().parent.parent  # tests/ -> repo root
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from typing import List, Optional, Set

import pytest

# from src.core.config import Settings # No longer importing real Settings


class MockSettings:  # Does NOT inherit from real Settings
    """Mock Settings class for testing."""

    debug: bool = False
    pipeline_version: str = "v_test_pipeline"
    commit_sha: Optional[str] = None
    prometheus_enabled: bool = True

    # API key configuration
    allowed_api_keys: List[str] = []

    # File upload settings
    # Define all fields that the application might access from settings
    allowed_extensions_raw: str = (
        "pdf,docx,xlsx,xls,csv,jpg,jpeg,png,tiff,tif,gif,bmp,eml,msg,txt"
    )
    allowed_extensions: Set[str] = {
        "pdf",
        "docx",
        "xlsx",
        "xls",
        "csv",
        "jpg",
        "jpeg",
        "png",
        "tiff",
        "tif",
        "gif",
        "bmp",
        "eml",
        "msg",
        "txt",
    }
    max_file_size_mb: int = 10
    max_batch_size: int = 50

    # Classification confidence settings
    confidence_threshold: float = 0.65
    early_exit_confidence: float = (
        0.90  # Ensure this is >= confidence_threshold by default
    )

    def __init__(self, **kwargs):
        """Initialize with optional overrides for any attribute."""
        # Set default values from class attributes first
        for key, value in self.__class__.__dict__.items():
            if not key.startswith("__") and not callable(value):
                setattr(self, key, value)

        # Then apply any kwargs, potentially validating critical ones
        temp_confidence = kwargs.get("confidence_threshold", self.confidence_threshold)
        temp_early_exit = kwargs.get(
            "early_exit_confidence", self.early_exit_confidence
        )

        if temp_early_exit < temp_confidence:
            raise ValueError("EARLY_EXIT_CONFIDENCE must be >= CONFIDENCE_THRESHOLD")

        for key, value in kwargs.items():
            setattr(self, key, value)

        # If allowed_extensions_raw is provided in kwargs, re-calculate allowed_extensions
        if "allowed_extensions_raw" in kwargs:
            raw_value = kwargs["allowed_extensions_raw"]
            if raw_value == "":
                self.allowed_extensions = set()
            elif raw_value:
                self.allowed_extensions = {
                    ext.strip().lower().lstrip(".")
                    for ext in raw_value.split(",")
                    if ext.strip()
                }
            else:  # raw_value is None
                self.allowed_extensions = (
                    set()
                )  # Or based on a default raw string if appropriate

    def is_extension_allowed(self, extension: str) -> bool:
        """Check if file extension is allowed."""
        if not extension:
            return False
        clean_ext = extension.lower().lstrip(".")
        return clean_ext in self.allowed_extensions


@pytest.fixture
def mock_settings():
    """Provide a plain MockSettings instance. Dependency injection is handled by app.dependency_overrides in client fixtures."""
    # REMOVED patch context managers.
    # The client fixture should use app.dependency_overrides exclusively for integration tests.
    settings = MockSettings()
    yield settings
    # Cleanup if needed, though typically not for a simple settings object.


@pytest.fixture(autouse=True)
def _disable_dotenv(monkeypatch):
    """Prevent the application Settings class from reading the developer *.env* file.*

    Unit-tests must operate against a *clean* environment.  Loading the real
    *.env* would inject values such as ``ALLOWED_EXTENSIONS`` that invalidate
    default-value assertions (see *tests/unit/core/test_config.py*).

    The fixture patches ``Settings.model_config['env_file']`` to ``None`` so
    that Pydantic skips dotenv processing entirely.  Individual tests remain
    free to manipulate environment variables via ``monkeypatch`` without having
    to worry about local developer configuration leaking in.
    """

    from src.core.config import Settings  # Imported here to avoid circularity

    # Disable reading of the external *.env* file for the entire test session.
    monkeypatch.setitem(Settings.model_config, "env_file", None)

    # Also guarantee that the two variables most often used in configuration
    # default-value tests are absent unless explicitly set by a test case.
    monkeypatch.delenv("ALLOWED_EXTENSIONS", raising=False)
    monkeypatch.delenv("ALLOWED_API_KEYS", raising=False)
