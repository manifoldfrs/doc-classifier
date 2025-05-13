from __future__ import annotations

import json
import os
from typing import List, Set

import pytest

from src.core.config import Settings, _parse_csv_str, get_settings


def test_settings_default_values(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that Settings have correct default values when no env vars are set."""
    # Clear any environment variables that might affect the test
    monkeypatch.delenv("ALLOWED_API_KEYS", raising=False)
    monkeypatch.delenv("ALLOWED_EXTENSIONS_RAW", raising=False)  # Target the raw field
    # Ensure any explicit ALLOWED_EXTENSIONS env var does not interfere with defaults
    monkeypatch.delenv("ALLOWED_EXTENSIONS", raising=False)
    monkeypatch.delenv("DEBUG", raising=False)
    monkeypatch.delenv("PROMETHEUS_ENABLED", raising=False)
    monkeypatch.delenv("PIPELINE_VERSION", raising=False)
    monkeypatch.delenv("COMMIT_SHA", raising=False)

    # Use the actual Settings class for default checking
    get_settings.cache_clear()  # Ensure fresh instance
    settings = get_settings()

    assert settings.debug is False
    assert settings.allowed_api_keys == []  # Default is empty list
    # Default raw value is "pdf,docx,csv,jpg,jpeg,png"
    expected_default_extensions = {"pdf", "docx", "csv", "jpg", "jpeg", "png"}
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
    # Remove any explicit ALLOWED_EXTENSIONS to test raw parsing exclusively
    monkeypatch.delenv("ALLOWED_EXTENSIONS", raising=False)
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
    get_settings.cache_clear()
    settings = get_settings()
    settings.allowed_extensions = {"pdf", "docx", "jpg"}  # Override for test

    assert settings.is_extension_allowed("pdf") is True
    assert settings.is_extension_allowed(".docx") is True  # Handles leading dot
    assert settings.is_extension_allowed("JPG") is True  # Case-insensitive
    assert settings.is_extension_allowed("txt") is False  # Not in the set
    assert settings.is_extension_allowed("") is False
    assert settings.is_extension_allowed("nodot") is False
    assert settings.is_extension_allowed(None) is False  # type: ignore[arg-type]


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
    # Remove any explicit ALLOWED_EXTENSIONS to test empty raw override
    monkeypatch.delenv("ALLOWED_EXTENSIONS", raising=False)
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
        monkeypatch.delenv("ALLOWED_EXTENSIONS", raising=False)
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
    monkeypatch.delenv("ALLOWED_EXTENSIONS", raising=False)
    monkeypatch.setenv("ALLOWED_EXTENSIONS_RAW", "custom1,custom2")
    get_settings.cache_clear()
    settings = get_settings()
    assert settings.allowed_extensions == {"custom1", "custom2"}


def test_settings_api_keys_already_json_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test ALLOWED_API_KEYS when it's already a JSON string in env."""
    json_keys = json.dumps(["json_key1", "json_key2"])
    monkeypatch.setenv("ALLOWED_API_KEYS", json_keys)
    get_settings.cache_clear()
    settings = get_settings()
    assert settings.allowed_api_keys == ["json_key1", "json_key2"]


def test_settings_extensions_already_json_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test ALLOWED_EXTENSIONS when it's already a JSON string in env."""
    json_exts = json.dumps(["json_pdf", "json_txt"])
    monkeypatch.setenv(
        "ALLOWED_EXTENSIONS", json_exts
    )  # This will be seen by _coerce_allowed_extensions
    monkeypatch.delenv(
        "ALLOWED_EXTENSIONS_RAW", raising=False
    )  # Ensure raw is not used
    get_settings.cache_clear()
    settings = get_settings()
    assert settings.allowed_extensions == {"json_pdf", "json_txt"}


def test_coerce_api_keys_with_none_and_json_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test _coerce_allowed_api_keys with None and JSON list string."""
    # Test with None
    monkeypatch.delenv("ALLOWED_API_KEYS", raising=False)
    get_settings.cache_clear()
    settings = Settings()  # Directly instantiate to test validator in isolation
    assert settings.allowed_api_keys == []

    # Test with JSON list string
    monkeypatch.setenv("ALLOWED_API_KEYS", '["key_json_1", "key_json_2"]')
    get_settings.cache_clear()
    settings = Settings()
    assert settings.allowed_api_keys == ["key_json_1", "key_json_2"]

    # Test with malformed JSON string (should parse as CSV)
    monkeypatch.setenv("ALLOWED_API_KEYS", '["key_malformed", key_also_mal')
    get_settings.cache_clear()
    settings = Settings()
    assert settings.allowed_api_keys == ['["key_malformed"', "key_also_mal"]


def test_coerce_extensions_with_none_empty_json_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test _coerce_allowed_extensions with various inputs."""
    # Test with ALLOWED_EXTENSIONS as None (env var not set)
    # and allowed_extensions_raw also None (passed to constructor)
    monkeypatch.delenv("ALLOWED_EXTENSIONS", raising=False)
    monkeypatch.delenv("ALLOWED_EXTENSIONS_RAW", raising=False)
    get_settings.cache_clear()
    settings = Settings(allowed_extensions_raw=None)
    assert settings.allowed_extensions == set()

    # Test with ALLOWED_EXTENSIONS as empty string
    monkeypatch.setenv("ALLOWED_EXTENSIONS", "")
    monkeypatch.delenv("ALLOWED_EXTENSIONS_RAW", raising=False)
    get_settings.cache_clear()
    settings = Settings()
    assert settings.allowed_extensions == set()

    # Test with ALLOWED_EXTENSIONS as JSON list string
    monkeypatch.setenv("ALLOWED_EXTENSIONS", '["ext_json1", ".ext_json2"]')
    monkeypatch.delenv("ALLOWED_EXTENSIONS_RAW", raising=False)
    get_settings.cache_clear()
    settings = Settings()
    assert settings.allowed_extensions == {"ext_json1", "ext_json2"}

    # Test with malformed JSON string (should parse as CSV)
    monkeypatch.setenv("ALLOWED_EXTENSIONS", '["ext_malformed", .ext_also_mal')
    monkeypatch.delenv("ALLOWED_EXTENSIONS_RAW", raising=False)
    get_settings.cache_clear()
    settings = Settings()
    assert settings.allowed_extensions == {'["ext_malformed"', "ext_also_mal"}


def test_settings_allowed_extensions_raw_is_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test parse_settings when allowed_extensions_raw is None."""
    monkeypatch.delenv(
        "ALLOWED_EXTENSIONS", raising=False
    )  # Ensure this doesn't interfere
    monkeypatch.delenv("ALLOWED_EXTENSIONS_RAW", raising=False)
    get_settings.cache_clear()
    # Instantiate settings directly to pass allowed_extensions_raw=None
    settings = Settings(allowed_extensions_raw=None)
    assert settings.allowed_extensions == set()


def test_settings_default_api_keys_when_env_var_is_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that allowed_api_keys is empty list if ALLOWED_API_KEYS env var is not set."""
    monkeypatch.delenv("ALLOWED_API_KEYS", raising=False)
    get_settings.cache_clear()
    settings = Settings()  # Instantiated with no env var for API keys
    assert settings.allowed_api_keys == []


def test_settings_default_extensions_when_env_var_is_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that allowed_extensions uses default from raw if ALLOWED_EXTENSIONS env var is not set."""
    monkeypatch.delenv("ALLOWED_EXTENSIONS", raising=False)  # Ensure this isn't set
    # ALLOWED_EXTENSIONS_RAW has a default value in the class definition
    default_raw_ext_val = "pdf,docx,csv,jpg,jpeg,png"
    expected_default_set = {
        ext.strip().lower().lstrip(".")
        for ext in default_raw_ext_val.split(",")
        if ext.strip()
    }

    get_settings.cache_clear()
    settings = Settings()  # Instantiated with no ALLOWED_EXTENSIONS env var
    assert settings.allowed_extensions == expected_default_set


def test_coerce_api_keys_direct_list_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test _coerce_allowed_api_keys when passed a list directly."""
    monkeypatch.delenv("ALLOWED_API_KEYS", raising=False)
    get_settings.cache_clear()
    settings = Settings(allowed_api_keys=["direct1", " direct2 ", "", 123])
    assert settings.allowed_api_keys == ["direct1", "direct2", "123"]


def test_coerce_extensions_direct_list_set_tuple(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test _coerce_allowed_extensions when passed list/set/tuple."""
    test_inputs = [
        ["list_ext1", ".list_ext2", ""],
        {"set_ext1", ".set_ext2", ""},
        ("tuple_ext1", ".tuple_ext2", ""),
    ]
    expected_outputs = [
        {"list_ext1", "list_ext2"},
        {"set_ext1", "set_ext2"},
        {"tuple_ext1", "tuple_ext2"},
    ]

    for direct_input, expected_set in zip(test_inputs, expected_outputs):
        monkeypatch.delenv("ALLOWED_EXTENSIONS", raising=False)
        monkeypatch.delenv("ALLOWED_EXTENSIONS_RAW", raising=False)
        get_settings.cache_clear()
        # Pass the list/set/tuple directly to the constructor
        settings = Settings(allowed_extensions=direct_input)
        assert (
            settings.allowed_extensions == expected_set
        ), f"Failed for input: {direct_input}"


def test_parse_settings_honors_explicit_empty_extensions_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test parse_settings honors ALLOWED_EXTENSIONS="" env var over raw default."""
    monkeypatch.setenv("ALLOWED_EXTENSIONS", "")
    # Keep the raw default present
    monkeypatch.setenv("ALLOWED_EXTENSIONS_RAW", "pdf,docx")
    get_settings.cache_clear()
    settings = Settings()
    assert settings.allowed_extensions == set()  # Explicit empty env var should win


def test_parse_settings_uses_raw_when_extensions_env_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test parse_settings falls back to raw when ALLOWED_EXTENSIONS env var is missing."""
    monkeypatch.delenv("ALLOWED_EXTENSIONS", raising=False)
    monkeypatch.setenv("ALLOWED_EXTENSIONS_RAW", "raw1, raw2")
    get_settings.cache_clear()
    settings = Settings()
    assert settings.allowed_extensions == {"raw1", "raw2"}


def test_coerce_api_keys_invalid_json(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test _coerce_allowed_api_keys falls back to CSV for invalid JSON."""
    invalid_json = '["key1", key2]'  # Missing quotes around key2
    monkeypatch.setenv("ALLOWED_API_KEYS", invalid_json)
    get_settings.cache_clear()
    settings = Settings()
    # Should parse as CSV, splitting the malformed string
    assert settings.allowed_api_keys == ['["key1"', "key2]"]


def test_coerce_extensions_invalid_json(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test _coerce_allowed_extensions falls back to CSV for invalid JSON."""
    invalid_json = '["ext1", .ext2]'  # Invalid syntax
    monkeypatch.setenv("ALLOWED_EXTENSIONS", invalid_json)
    monkeypatch.delenv("ALLOWED_EXTENSIONS_RAW", raising=False)
    get_settings.cache_clear()
    settings = Settings()
    assert settings.allowed_extensions == {'["ext1"', "ext2]"}
