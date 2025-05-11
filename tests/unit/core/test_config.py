"""tests/unit/core/test_config.py
###############################################################################
Unit tests for the application configuration model (``src.core.config.Settings``).
###############################################################################
These tests verify:
- Default values of settings.
- Parsing of raw string environment variables into structured properties.
- Behavior of helper methods like `is_extension_allowed`.
- Custom Pydantic validators like `validate_confidence_thresholds`.
- Functionality of the `get_settings` accessor.
"""

from __future__ import annotations

import pytest

from src.core.config import Settings, _parse_csv_str, get_settings


def test_settings_default_values(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that Settings have correct default values when no env vars are set."""
    # Clear any environment variables that might affect the test
    monkeypatch.delenv("ALLOWED_API_KEYS", raising=False)
    monkeypatch.delenv("ALLOWED_EXTENSIONS_RAW", raising=False)  # Target the raw field

    # Use the actual Settings class for default checking
    get_settings.cache_clear()  # Ensure fresh instance
    settings = get_settings()

    assert settings.debug is False
    assert settings.allowed_api_keys == []  # Default is empty list
    # Default extensions are defined in the Settings class itself
    expected_default_extensions = {
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
    assert settings.allowed_extensions == expected_default_extensions
    assert settings.max_file_size_mb == 10
    assert settings.max_batch_size == 50
    assert settings.confidence_threshold == 0.65
    assert settings.early_exit_confidence == 0.95
    assert settings.pipeline_version == "v0.1.0"  # Default from Settings class
    assert settings.prometheus_enabled is True
    assert settings.commit_sha is None


def test_settings_parsing_from_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that Settings correctly parse values from environment variables."""
    monkeypatch.setenv("DEBUG", "true")
    monkeypatch.setenv("ALLOWED_API_KEYS", "key1, key2, key3")
    monkeypatch.setenv("ALLOWED_EXTENSIONS_RAW", "py, .Js, TXT")
    monkeypatch.setenv("MAX_FILE_SIZE_MB", "20")
    monkeypatch.setenv("CONFIDENCE_THRESHOLD", "0.77")
    monkeypatch.setenv("PIPELINE_VERSION", "v1.2.3-env")
    monkeypatch.setenv("COMMIT_SHA", "testsha123env")
    monkeypatch.setenv("PROMETHEUS_ENABLED", "false")

    get_settings.cache_clear()  # Reload settings from new env
    settings = get_settings()

    assert settings.debug is True
    assert settings.allowed_api_keys == ["key1", "key2", "key3"]

    expected_extensions = {"py", "js", "txt"}
    assert settings.allowed_extensions == expected_extensions

    assert settings.max_file_size_mb == 20
    assert settings.confidence_threshold == 0.77
    assert settings.pipeline_version == "v1.2.3-env"
    assert settings.commit_sha == "testsha123env"
    assert settings.prometheus_enabled is False


def test_settings_is_extension_allowed() -> None:
    """Test the `is_extension_allowed` helper method."""
    settings = get_settings()  # Use actual settings for this helper
    settings.allowed_extensions = {"pdf", "docx", "jpg"}  # Override for test

    assert settings.is_extension_allowed("pdf") is True
    assert settings.is_extension_allowed(".docx") is True  # Handles leading dot
    assert settings.is_extension_allowed("JPG") is True  # Case-insensitive
    assert settings.is_extension_allowed("txt") is False  # Not in the set
    assert settings.is_extension_allowed("") is False
    assert settings.is_extension_allowed("nodot") is False
    assert settings.is_extension_allowed(None) is False


def test_settings_confidence_threshold_validator_valid() -> None:
    """Test the confidence threshold validator with valid inputs."""
    # This uses the actual Settings class and its validators
    get_settings.cache_clear()
    try:
        Settings(confidence_threshold=0.5, early_exit_confidence=0.6)
        Settings(confidence_threshold=0.7, early_exit_confidence=0.7)
    except ValueError:
        pytest.fail("ValueError raised unexpectedly for valid confidence thresholds.")


def test_settings_confidence_threshold_validator_invalid() -> None:
    """Test the confidence threshold validator with invalid inputs."""
    get_settings.cache_clear()
    with pytest.raises(ValueError) as exc_info:
        Settings(confidence_threshold=0.8, early_exit_confidence=0.7)
    assert "EARLY_EXIT_CONFIDENCE must be >= CONFIDENCE_THRESHOLD" in str(
        exc_info.value
    )


def test_get_settings_returns_settings_instance() -> None:
    """Test that get_settings() returns an instance of Settings."""
    get_settings.cache_clear()
    settings = get_settings()
    assert isinstance(settings, Settings)


def test_get_settings_caching_normal_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that get_settings() caches the Settings instance in non-pytest environments."""
    monkeypatch.delenv(
        "PYTEST_CURRENT_TEST", raising=False
    )  # Ensure not in pytest test run mode for this
    get_settings.cache_clear()

    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2


def test_get_settings_pytest_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that get_settings() returns a fresh instance when PYTEST_CURRENT_TEST is set."""
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "some_test_is_running")
    get_settings.cache_clear()  # Clear cache before test

    s1 = get_settings()
    s2 = get_settings()  # Should be a new instance
    assert s1 is not s2
    monkeypatch.delenv("PYTEST_CURRENT_TEST")  # Clean up


def test_allowed_extensions_empty_string_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test ALLOWED_EXTENSIONS_RAW being an empty string from env."""
    monkeypatch.setenv("ALLOWED_EXTENSIONS_RAW", "")
    get_settings.cache_clear()
    settings = get_settings()
    assert settings.allowed_extensions == set()  # Should parse to empty set
    assert settings.is_extension_allowed("pdf") is False


def test_allowed_api_keys_empty_string_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test ALLOWED_API_KEYS being an empty string from env."""
    monkeypatch.setenv("ALLOWED_API_KEYS", "")
    get_settings.cache_clear()
    settings = get_settings()
    assert settings.allowed_api_keys == []


def test_allowed_api_keys_with_whitespace_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test ALLOWED_API_KEYS from env with keys having leading/trailing whitespace."""
    monkeypatch.setenv("ALLOWED_API_KEYS", " key1 , key2  ,  key3 ")
    get_settings.cache_clear()
    settings = get_settings()
    assert settings.allowed_api_keys == ["key1", "key2", "key3"]


def test_settings_allowed_extensions_parsing(monkeypatch: pytest.MonkeyPatch):
    """Test various formats for ALLOWED_EXTENSIONS_RAW."""
    test_cases = {
        "pdf,docx,.jpg": {"pdf", "docx", "jpg"},
        " TXT , md ": {"txt", "md"},
        ".Zip": {"zip"},
        "": set(),
        "json": {"json"},
    }
    for raw_value, expected_set in test_cases.items():
        monkeypatch.setenv("ALLOWED_EXTENSIONS_RAW", raw_value)
        get_settings.cache_clear()
        settings = get_settings()
        assert (
            settings.allowed_extensions == expected_set
        ), f"Failed for input: '{raw_value}'"


def test_settings_api_keys_parsing_from_env(monkeypatch: pytest.MonkeyPatch):
    """Test parsing of ALLOWED_API_KEYS from env with comma-separated string."""
    monkeypatch.setenv("ALLOWED_API_KEYS", "keyA,keyB,keyC")
    get_settings.cache_clear()
    settings = get_settings()
    assert settings.allowed_api_keys == ["keyA", "keyB", "keyC"]

    monkeypatch.setenv("ALLOWED_API_KEYS", "singlekey")
    get_settings.cache_clear()
    settings = get_settings()
    assert settings.allowed_api_keys == ["singlekey"]


def test_parse_csv_str_helper():
    """Test the _parse_csv_str helper function directly."""
    assert _parse_csv_str("a,b,c") == ["a", "b", "c"]
    assert _parse_csv_str(" a , b , c ") == ["a", "b", "c"]
    assert _parse_csv_str("single") == ["single"]
    assert _parse_csv_str("") == []
    assert _parse_csv_str(" , ") == []  # Only separators
    assert _parse_csv_str("a,,b") == ["a", "b"]  # Empty elements


def test_settings_default_extensions_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that ALLOWED_EXTENSIONS_RAW overrides default extensions."""
    monkeypatch.setenv("ALLOWED_EXTENSIONS_RAW", "custom1,custom2")
    get_settings.cache_clear()
    settings = get_settings()
    assert settings.allowed_extensions == {"custom1", "custom2"}
