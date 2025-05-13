from __future__ import annotations

import uuid
from io import BytesIO
from typing import List
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException, UploadFile, status
from fastapi.testclient import TestClient

from src.api.app import app
from src.core.config import get_settings
from tests.conftest import MockSettings

pytestmark = [pytest.mark.integration]


def _build_multipart_payload(
    count: int = 3, include_unsupported_ext: bool = False
) -> List[tuple[str, tuple[str, BytesIO, str]]]:
    """Return a **files** payload suitable for ``client.post(..., files=...)``.

    Args:
        count: Number of "valid" text files to generate.
        include_unsupported_ext: Whether to add one file with a .zip extension.
    """
    samples = []
    for i in range(count):
        samples.append(
            (f"test_file_{i+1}.txt", f"Content of file {i+1}".encode(), "text/plain")
        )

    if include_unsupported_ext:
        samples.append(("unsupported.zip", b"zip content", "application/zip"))

    return [
        ("files", (name, BytesIO(content), mime)) for name, content, mime in samples
    ]


@pytest.fixture
def client(mock_settings: MockSettings) -> TestClient:
    """Provides a TestClient instance with dependency overrides."""
    # Override the settings dependency for the entire app
    app.dependency_overrides[get_settings] = lambda: mock_settings
    # Ensure these are set for the client fixture, tests can override if needed
    mock_settings.allowed_api_keys = ["test-api-key"]
    mock_settings.allowed_extensions = {"txt"}
    yield TestClient(app)
    # Clean up overrides after test
    app.dependency_overrides = {}


@pytest.fixture
def headers(mock_settings: MockSettings) -> dict[str, str]:
    """Provides default headers including the API key and a test request ID."""
    # Ensure allowed_api_keys is populated before trying to access it
    if not mock_settings.allowed_api_keys:
        mock_settings.allowed_api_keys = ["test-api-key-default"]
    return {
        "x-api-key": mock_settings.allowed_api_keys[0],
        "X-Request-ID": f"test-req-id-{uuid.uuid4().hex}",
    }


def test_batch_upload_three_files_returns_expected_shape(
    client: TestClient, mock_settings: MockSettings, headers: dict[str, str]
) -> None:
    """Upload 3 files and assert the synchronous 200-OK JSON structure.

    The endpoint should return **200 OK** (since 3 < ASYNC_THRESHOLD=10) and a
    JSON array of length 3 where each element contains at least the following
    keys mandated by the spec: ``filename``, ``label``, ``confidence``.
    """
    # mock_settings is already configured by the client fixture
    # headers fixture provides the necessary X-Request-ID

    # Patch classify where it's used: in src.api.routes.files
    with patch(
        "src.api.routes.files.classify", new_callable=AsyncMock
    ) as mock_classify:

        class _StubResult(dict):
            def dict(self):
                return self

        async def _fake_classify(file: UploadFile):
            # Read content for size calculation and then reset pointer
            content_bytes = await file.read()
            await file.seek(0)
            return _StubResult(
                filename=file.filename,
                label="unknown",
                confidence=0.0,
                mime_type=file.content_type or "text/plain",
                size_bytes=len(content_bytes),
                pipeline_version=mock_settings.pipeline_version,
                processing_ms=0.0,
                warnings=[],
                errors=[],
                # request_id will be set by the route based on header
            )

        mock_classify.side_effect = _fake_classify

        response = client.post(
            "/v1/files", files=_build_multipart_payload(count=3), headers=headers
        )

        assert response.status_code == 200, response.text
        assert response.headers.get("content-type", "").startswith("application/json")

        payload = response.json()
        assert isinstance(payload, list)
        assert len(payload) == 3

        expected_request_id = headers["X-Request-ID"]
        assert response.headers.get("X-Request-ID") == expected_request_id

        required_keys = {"filename", "label", "confidence", "request_id"}
        for item in payload:
            assert isinstance(item, dict)
            assert required_keys.issubset(item.keys())
            assert isinstance(item["label"], str)
            assert 0.0 <= float(item["confidence"]) <= 1.0
            assert item["request_id"] == expected_request_id
        mock_classify.assert_called()  # Check if the mock was called
        assert mock_classify.call_count == 3  # Should be called for each file


def test_upload_batch_exceeds_limit(
    client: TestClient, headers: dict[str, str], mock_settings: MockSettings
) -> None:
    """Test submitting more files than MAX_BATCH_SIZE returns 413."""
    mock_settings.max_batch_size = 2
    # The client fixture already sets the dependency override.
    # No need to set app.dependency_overrides[get_settings] = lambda: mock_settings here again.

    files_payload = _build_multipart_payload(count=3)  # Generate 3 files

    response = client.post("/v1/files", files=files_payload, headers=headers)

    assert response.status_code == 413
    payload = response.json()
    assert payload["error"]["code"] == 413
    # The route dynamically creates the message based on len(files)
    assert payload["error"]["message"] == "Batch size 3 exceeds limit of 2."
    assert payload["detail"] == "Batch size 3 exceeds limit of 2."


def test_upload_with_unsupported_extension(
    client: TestClient, headers: dict[str, str], mock_settings: MockSettings
) -> None:
    """Test submitting a file with an unsupported extension returns 415."""
    # client fixture sets allowed_extensions to {"txt"}
    # _build_multipart_payload by default creates .txt files
    # We need one .zip file
    files_payload = _build_multipart_payload(count=1, include_unsupported_ext=True)

    response = client.post("/v1/files", files=files_payload, headers=headers)

    assert response.status_code == 415
    assert "Unsupported file extension: .zip" in response.json()["detail"]


def test_upload_no_files(
    client: TestClient, headers: dict[str, str], mock_settings: MockSettings
) -> None:
    """Test submitting the form with no files attached (empty list for 'files' field) returns 422."""
    # This test now expects a 422 because List[UploadFile] = File(...)
    # implies at least one file is required by FastAPI/Pydantic validation.
    response = client.post("/v1/files", files=[], headers=headers)

    assert response.status_code == 422  # Changed from 400
    payload = response.json()
    assert "error" in payload
    assert payload["error"]["code"] == "validation_error"
    assert "details" in payload["error"]
    assert any(
        "files" in detail.get("loc", []) for detail in payload["error"]["details"]
    )


def test_upload_no_files_field_fastapi_validation(
    client: TestClient, headers: dict[str, str]
) -> None:
    """
    Test sending a POST request where the 'files' multipart field is entirely missing.
    FastAPI's RequestValidationError should be caught by our custom handler.
    """
    response = client.post("/v1/files", data={"other_field": "value"}, headers=headers)

    assert response.status_code == 422
    payload = response.json()
    assert "error" in payload
    assert payload["error"]["code"] == "validation_error"
    assert payload["error"]["message"] == "Invalid request parameters."
    assert "details" in payload["error"]
    assert isinstance(payload["error"]["details"], list)
    assert len(payload["error"]["details"]) > 0
    detail_item = payload["error"]["details"][0]
    # Example of how FastAPI/Pydantic might report a missing 'files' field
    # This can vary slightly based on Pydantic/FastAPI versions
    assert detail_item["type"] == "missing" or "field_required" in detail_item["type"]
    assert "files" in detail_item["loc"]  # Ensure 'files' is part of the location
    assert (
        "Field required" in detail_item["msg"]
        or "field required" in detail_item["msg"].lower()
    )


def test_upload_empty_files_list(client: TestClient, headers: dict[str, str]) -> None:
    """
    Test sending a POST request where the 'files' field is present but empty.
    This should also result in a RequestValidationError caught by our custom handler.
    """
    response = client.post("/v1/files", files=[], headers=headers)

    assert response.status_code == 422
    payload = response.json()
    assert "error" in payload
    assert payload["error"]["code"] == "validation_error"
    assert payload["error"]["message"] == "Invalid request parameters."
    assert "details" in payload["error"]
    assert isinstance(payload["error"]["details"], list)
    assert any(
        "files" in detail.get("loc", []) for detail in payload["error"]["details"]
    ), "Error detail should reference the 'files' field."
    assert any(
        "field required" in detail.get("msg", "").lower()
        or "ensure this value has at least 1 item" in detail.get("msg", "").lower()
        or "list should have at least 1 item" in detail.get("msg", "").lower()
        for detail in payload["error"]["details"]
    ), "Error message should indicate missing or insufficient files for the 'files' field."


def test_upload_with_validation_error_raised_by_route_validator(
    client: TestClient, headers: dict[str, str], mock_settings: MockSettings
) -> None:
    """Test the scenario where src.ingestion.validators.validate_file raises an HTTPException."""
    # This test will send one valid .txt file according to mock_settings in client fixture
    files_payload = _build_multipart_payload(count=1)

    validation_exception = HTTPException(
        status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
        detail="Custom validator error: Unsupported file extension: .bad",
    )
    # Mock the validate_file function imported in src.api.routes.files
    with patch(
        "src.api.routes.files.validate_file", side_effect=validation_exception
    ) as mock_validate_file_in_route:
        response = client.post("/v1/files", files=files_payload, headers=headers)

        assert response.status_code == 415
        payload = response.json()
        # This HTTPException is caught by _http_exception_handler
        assert payload["error"]["code"] == 415
        assert (
            payload["error"]["message"]
            == "Custom validator error: Unsupported file extension: .bad"
        )
        assert (
            payload["detail"]
            == "Custom validator error: Unsupported file extension: .bad"
        )
        # Ensure validate_file was called for each file. Here, it's one file.
        assert mock_validate_file_in_route.call_count == 1
