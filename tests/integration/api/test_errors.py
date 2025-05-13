from __future__ import annotations

import json

import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field

from src.api.app import app as main_app  # Import the main app instance
from src.api.errors import (
    _http_exception_handler,
    _unhandled_exception_handler,
    _validation_error_handler,
)
from src.core.config import get_settings
from tests.conftest import MockSettings

pytestmark = [pytest.mark.integration]


class DummyModel(BaseModel):
    name: str = Field(..., min_length=3)


@pytest.fixture
def test_app(mock_settings: MockSettings) -> FastAPI:
    """Creates a minimal FastAPI app instance for testing error handlers directly."""
    # Override settings dependency for the main app to ensure consistency if needed
    main_app.dependency_overrides[get_settings] = lambda: mock_settings

    # You might not need a full app instance if you test handlers directly,
    # but it's useful for simulating requests.
    temp_app = FastAPI()

    # Add handlers to the temporary app instance if testing via TestClient
    temp_app.add_exception_handler(RequestValidationError, _validation_error_handler)
    temp_app.add_exception_handler(HTTPException, _http_exception_handler)
    temp_app.add_exception_handler(Exception, _unhandled_exception_handler)

    @temp_app.post("/validation")
    async def cause_validation_error(item: DummyModel):
        return {"name": item.name}

    @temp_app.get("/http_exception")
    async def cause_http_exception():
        raise HTTPException(status_code=403, detail="Forbidden access")

    @temp_app.get("/unhandled_exception")
    async def cause_unhandled_exception():
        raise ValueError("Something unexpected went wrong")

    return temp_app


@pytest.fixture
def client(test_app: FastAPI, mock_settings: MockSettings) -> TestClient:
    """Provides a TestClient instance configured for error testing."""
    # Ensure the test app also uses mocked settings if needed by handlers
    test_app.dependency_overrides[get_settings] = lambda: mock_settings
    # Important: raise_server_exceptions=False to test actual handler responses for 500 errors
    return TestClient(test_app, raise_server_exceptions=False)


def test_validation_error_handler(client: TestClient) -> None:
    """Test the 422 response for Pydantic validation errors."""
    response = client.post("/validation", json={"name": "a"})

    assert response.status_code == 422
    payload = response.json()
    assert "error" in payload
    assert payload["error"]["code"] == "validation_error"
    assert payload["error"]["message"] == "Invalid request parameters."
    assert "details" in payload["error"]
    assert isinstance(payload["error"]["details"], list)
    assert len(payload["error"]["details"]) > 0
    assert payload["error"]["details"][0]["loc"] == ["body", "name"]
    # Updated Pydantic v2 error message
    assert (
        "String should have at least 3 characters"
        in payload["error"]["details"][0]["msg"]
    )


def test_http_exception_handler(client: TestClient) -> None:
    """Test the response for standard FastAPI/Starlette HTTPErrors."""
    response = client.get("/http_exception")

    assert response.status_code == 403
    payload = response.json()
    assert "error" in payload
    assert payload["error"]["code"] == 403
    assert payload["error"]["message"] == "Forbidden access"
    # The specific handler also adds a top-level 'detail' key
    assert payload["detail"] == "Forbidden access"


def test_unhandled_exception_handler(client: TestClient) -> None:
    """Test the 500 response for generic, unhandled exceptions."""
    # We need to prevent the TestClient from raising the actual exception
    # and instead allow the handler to catch it. This usually works by default.
    response = client.get("/unhandled_exception")

    assert response.status_code == 500
    payload = response.json()
    assert "error" in payload
    assert payload["error"]["code"] == "internal_server_error"
    assert payload["error"]["message"] == "An unexpected error occurred."


@pytest.mark.asyncio
async def test_handler_direct_invocation(mock_request_fixture) -> None:
    """Example: Test a handler function directly (less common for integration)."""
    exception = ValueError("Direct test error")
    response = await _unhandled_exception_handler(mock_request_fixture, exception)

    assert response.status_code == 500
    # Corrected: JSONResponse.body is bytes, needs decoding then parsing
    payload = json.loads(response.body.decode())
    assert payload["error"]["code"] == "internal_server_error"
    assert payload["error"]["request_id"] == "direct-test-id"


@pytest.fixture
def mock_request_fixture(mock_settings: MockSettings) -> Request:
    """Provides a mock FastAPI Request object for direct handler tests."""
    # Correctly includes headers in the scope
    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "GET",
        "path": "/test/direct",
        # Headers should be list of tuples (byte string name, byte string value)
        "headers": [(b"x-request-id", b"direct-test-id")],
        "client": ("127.0.0.1", 8080),
        "server": ("localhost", 8000),
        "query_string": b"",
        "root_path": "",
        "scheme": "http",
        "app": FastAPI(),
        "state": {},
    }
    request = Request(scope)
    # Manually set the user state if needed for the test, as middleware won't run
    request.state.user = None
    return request
