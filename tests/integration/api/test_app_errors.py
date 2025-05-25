from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.api.app import _create_fastapi_app
from src.api.errors import (
    _http_exception_handler,
    _unhandled_exception_handler,
    _validation_error_handler,
    add_exception_handlers,
)
from src.core.config import Settings, get_settings
from tests.conftest import MockSettings


@pytest.fixture
def mock_request_scope() -> dict:
    """Provides a mock ASGI scope for Request."""
    return {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "server": ("testserver", 80),
        "client": ("testclient", 123),
        "scheme": "http",
        "method": "GET",
        "root_path": "",
        "path": "/test",
        "raw_path": b"/test",
        "query_string": b"",
        "headers": [],
    }


@pytest.mark.asyncio
async def test_http_exception_handler_generic_exception(
    mock_request_scope: dict,
) -> None:
    """Test _http_exception_handler with a generic Exception."""
    request = Request(mock_request_scope)
    exc = Exception("Generic test error")
    response = await _http_exception_handler(request, exc)
    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    payload = json.loads(response.body.decode())
    assert payload["error"]["message"] == "Generic test error"
    assert payload["error"]["code"] == 500
    assert payload["detail"] == "Generic test error"


@pytest.mark.asyncio
async def test_validation_error_handler_generic_exception(
    mock_request_scope: dict,
) -> None:
    """Test _validation_error_handler with a generic Exception."""
    request = Request(mock_request_scope)
    exc = Exception("Generic validation error")  # Simulate a non-RequestValidationError
    response = await _validation_error_handler(request, exc)
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    payload = json.loads(response.body.decode())
    assert payload["error"]["message"] == "Invalid request parameters."
    assert payload["error"]["code"] == "validation_error"
    assert "Generic validation error" in str(payload["error"]["details"])


@pytest.mark.asyncio
async def test_unhandled_exception_handler(mock_request_scope: dict) -> None:
    """Test the _unhandled_exception_handler."""
    request = Request(mock_request_scope)
    exc = ValueError("Something went very wrong")
    response = await _unhandled_exception_handler(request, exc)
    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    payload = json.loads(response.body.decode())
    assert payload["error"]["message"] == "An unexpected error occurred."
    assert payload["error"]["code"] == "internal_server_error"


def test_add_exception_handlers() -> None:
    """Test that add_exception_handlers registers handlers."""
    app_instance = FastAPI()
    add_exception_handlers(app_instance)
    assert len(app_instance.exception_handlers) > 0
    # Check for specific handlers by type if necessary
    assert StarletteHTTPException in app_instance.exception_handlers
    assert RequestValidationError in app_instance.exception_handlers
    assert Exception in app_instance.exception_handlers


def test_prometheus_disabled_package_available(mock_settings: MockSettings) -> None:
    """Test app creation when Prometheus is disabled but package is available."""
    mock_settings.prometheus_enabled = False
    with patch("src.api.app.settings", mock_settings):
        with patch("src.api.app._PROM_AVAILABLE", True):  # Simulate package is there
            test_app = _create_fastapi_app()
            client = TestClient(test_app)
            response = client.get("/metrics")  # Should be 404 if not exposed
            assert response.status_code == 404


def test_prometheus_enabled_package_missing(mock_settings: MockSettings) -> None:
    """Test app creation when Prometheus is enabled but package is missing."""
    mock_settings.prometheus_enabled = True
    with patch("src.api.app.settings", mock_settings):
        with patch("src.api.app._PROM_AVAILABLE", False):  # Simulate package is missing
            with patch("src.api.app.logger") as mock_logger:
                test_app = _create_fastapi_app()
                client = TestClient(test_app)
                response = client.get("/metrics")  # Should be 404
                assert response.status_code == 404
                mock_logger.warning.assert_called_with(
                    "prometheus_instrumentation_requested_but_package_missing",
                    advice="Add 'prometheus-fastapi-instrumentator' to requirements.txt",
                )


def test_root_endpoint(mock_settings: MockSettings) -> None:
    """Test the root GET / endpoint."""
    with patch("src.api.app.settings", mock_settings):
        test_app = _create_fastapi_app()
        client = TestClient(test_app)
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"message": "Document Classifier – FastAPI layer"}


def test_app_startup_shutdown_logs(mock_settings: MockSettings):
    mock_settings.commit_sha = "testcommit123"
    with patch("src.api.app.settings", mock_settings):
        with patch("src.api.app.logger") as mock_logger:
            test_app_instance = _create_fastapi_app()
            with TestClient(test_app_instance) as client:  # noqa F841
                pass  # Trigger startup/shutdown

            startup_logged = any(
                call.args[0] == "fastapi_startup"
                for call in mock_logger.info.call_args_list
            )
            shutdown_logged = any(
                call.args[0] == "fastapi_shutdown"
                for call in mock_logger.info.call_args_list
            )
            assert startup_logged
            assert shutdown_logged

            startup_call_kwargs = next(
                (
                    call.kwargs
                    for call in mock_logger.info.call_args_list
                    if call.args[0] == "fastapi_startup"
                ),
                None,
            )
            assert startup_call_kwargs is not None
            assert startup_call_kwargs.get("commit_sha") == "testcommit123"
