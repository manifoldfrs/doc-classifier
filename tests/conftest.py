# ruff: noqa: E402
from __future__ import annotations

import sys
from pathlib import Path

# Ensure repository root is first on sys.path
_repo_root: Path = Path(__file__).resolve().parent.parent  # tests/ -> repo root
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from typing import List, Optional, Set
from unittest.mock import patch

import pytest


class MockSettings:
    """Mock Settings class for testing."""

    debug: bool = False
    pipeline_version: str = "v_test_pipeline"
    commit_sha: Optional[str] = None
    prometheus_enabled: bool = True

    # API key configuration
    allowed_api_keys: List[str] = []

    # File upload settings
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
    early_exit_confidence: float = 0.9

    def __init__(self, **kwargs):
        """Initialize with optional overrides."""
        for key, value in kwargs.items():
            if key == "allowed_extensions_raw":
                # Handle special case for extensions
                if value == "":
                    self.allowed_extensions = set()
                elif value:
                    self.allowed_extensions = {
                        ext.strip().lower().lstrip(".")
                        for ext in value.split(",")
                        if ext.strip()
                    }
            elif key == "confidence_threshold" and "early_exit_confidence" in kwargs:
                # Check for valid confidence thresholds
                if kwargs["early_exit_confidence"] < value:
                    raise ValueError(
                        "EARLY_EXIT_CONFIDENCE must be >= CONFIDENCE_THRESHOLD"
                    )
            setattr(self, key, value)

    def is_extension_allowed(self, extension: str) -> bool:
        """Check if file extension is allowed."""
        if not extension:
            return False
        clean_ext = extension.lower().lstrip(".")
        return clean_ext in self.allowed_extensions


@pytest.fixture
def mock_settings():
    """Provide a mock Settings instance for tests."""
    # Patch **both** locations where get_settings is imported so that
    # every part of the application (including auth dependencies) receives
    # the exact same MockSettings instance.  This avoids discrepancies where
    # a previously-imported alias still points to the original function.
    with (
        patch("src.core.config.get_settings") as mock_get_core_settings,
        patch("src.utils.auth.get_settings") as mock_get_auth_settings,
    ):
        settings = MockSettings()
        mock_get_core_settings.return_value = settings
        mock_get_auth_settings.return_value = settings
        yield settings


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
