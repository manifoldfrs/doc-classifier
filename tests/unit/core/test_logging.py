from __future__ import annotations

import logging
from unittest.mock import patch, ANY

import pytest
import structlog

from src.core.logging import _LOGGING_CONFIGURED, configure_logging
import src.core.logging  # Added for monkeypatching


@pytest.fixture(autouse=True)
def reset_logging_state(monkeypatch: pytest.MonkeyPatch):
    """Ensure logging configuration state is reset before and after each test."""
    # Use monkeypatch to modify the _LOGGING_CONFIGURED in the source module directly
    original_state = getattr(src.core.logging, "_LOGGING_CONFIGURED", False)
    monkeypatch.setattr("src.core.logging._LOGGING_CONFIGURED", False)
    structlog.reset_defaults()  # Reset structlog's internal state
    # Clear any handlers that might have been added to the root logger
    root_logger = logging.getLogger()
    # Store original handlers
    original_handlers = root_logger.handlers[:]
    root_logger.handlers.clear()

    yield

    monkeypatch.setattr("src.core.logging._LOGGING_CONFIGURED", original_state)
    structlog.reset_defaults()
    # Restore original handlers
    root_logger.handlers.clear()
    for handler in original_handlers:
        root_logger.addHandler(handler)


def test_configure_logging_idempotency():
    """Test that configure_logging is idempotent and only configures once."""
    with (
        patch("src.core.logging._configure_stdlib_logging") as mock_stdlib_config,
        patch("structlog.configure") as mock_structlog_config,
    ):

        # First call - should configure
        configure_logging(debug=True)
        assert _LOGGING_CONFIGURED is True
        mock_stdlib_config.assert_called_once_with(logging.DEBUG)
        mock_structlog_config.assert_called_once()

        # Reset mock call counts for the second call check
        mock_stdlib_config.reset_mock()
        mock_structlog_config.reset_mock()

        # Second call - should be a no-op
        configure_logging(debug=False)  # debug flag should not matter now
        assert _LOGGING_CONFIGURED is True  # Still true
        mock_stdlib_config.assert_not_called()
        mock_structlog_config.assert_not_called()


def test_configure_logging_sets_debug_level():
    """Test that configure_logging sets the correct log level for debug=True."""
    with (
        patch("src.core.logging._configure_stdlib_logging") as mock_stdlib_config,
        patch("structlog.make_filtering_bound_logger") as mock_make_filtering_logger,
        patch("structlog.configure") as mock_structlog_config,
    ):
        # We need mock_structlog_config to ensure the overall configure path is tested,
        # but we assert against mock_make_filtering_logger for the level.

        configure_logging(debug=True)

        mock_stdlib_config.assert_called_once_with(logging.DEBUG)
        mock_make_filtering_logger.assert_called_once_with(logging.DEBUG)
        # Ensure structlog.configure was called, with the result of make_filtering_bound_logger
        mock_structlog_config.assert_called_once()
        assert (
            mock_structlog_config.call_args[1]["wrapper_class"]
            == mock_make_filtering_logger.return_value
        )


def test_configure_logging_sets_info_level():
    """Test that configure_logging sets the correct log level for debug=False."""
    with (
        patch("src.core.logging._configure_stdlib_logging") as mock_stdlib_config,
        patch("structlog.make_filtering_bound_logger") as mock_make_filtering_logger,
        patch("structlog.configure") as mock_structlog_config,
    ):

        configure_logging(debug=False)

        mock_stdlib_config.assert_called_once_with(logging.INFO)
        mock_make_filtering_logger.assert_called_once_with(logging.INFO)
        mock_structlog_config.assert_called_once()
        assert (
            mock_structlog_config.call_args[1]["wrapper_class"]
            == mock_make_filtering_logger.return_value
        )
