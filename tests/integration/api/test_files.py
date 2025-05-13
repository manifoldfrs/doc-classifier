from __future__ import annotations

import uuid
from io import BytesIO
from typing import List
from unittest.mock import patch

import pytest
from fastapi import HTTPException, UploadFile, status
from fastapi.testclient import TestClient

from src.api.app import app
from src.core.config import get_settings
from tests.conftest import MockSettings

pytestmark = [pytest.mark.integration]


def _build_multipart_payload(
    count: int = 3,
) -> List[tuple[str, tuple[str, BytesIO, str]]]:
    """Return a **files** payload suitable for ``client.post(..., files=...)``.

    Generates `count` simple text files.
    """
    samples = [
        (f"file_{i}.txt", f"content {i}".encode("utf-8"), "text/plain")
        for i in range(count)
    ]
    return [
        ("files", (name, BytesIO(content), mime)) for name, content, mime in samples
    ]


@pytest.fixture
def client(mock_settings: MockSettings) -> TestClient:
    """Provides a TestClient instance with overridden settings for file tests."""
    app.dependency_overrides[get_settings] = lambda: mock_settings
    mock_settings.allowed_api_keys = ["test-api-key"]
    # Ensure TestClient does not raise server exceptions for 500 error testing if any
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def headers(mock_settings: MockSettings) -> dict[str, str]:
    """Provides default headers including the API key and a test request ID."""
    return {
        "x-api-key": mock_settings.allowed_api_keys[0],
        "X-Request-ID": f"test-req-id-{uuid.uuid4().hex}",
    }


def test_batch_upload_three_files_returns_expected_shape(
    client: TestClient, headers: dict[str, str], mock_settings: MockSettings
) -> None:
    """Upload 3 files and assert the synchronous 200-OK JSON structure."""

    with (
        patch("src.core.config.get_settings", return_value=mock_settings),
        patch("src.utils.auth.get_settings", return_value=mock_settings),
        patch("src.api.routes.files.classify") as mock_classify,
        patch("src.api.routes.files.validate_file", return_value=None) as mock_validate,
    ):

        class MockInternalClassificationResult:
            def __init__(
                self,
                filename,
                mime_type,
                size_bytes,
                label,
                confidence,
                pipeline_version,
                processing_ms,
                warnings,
                errors,
            ):
                self.filename = filename
                self.mime_type = mime_type
                self.size_bytes = size_bytes
                self.label = label
                self.confidence = confidence
                self.stage_confidences = {}
                self.pipeline_version = pipeline_version
                self.processing_ms = processing_ms
                self.warnings = warnings
                self.errors = errors

            def dict(self):
                return {
                    "filename": self.filename,
                    "mime_type": self.mime_type,
                    "size_bytes": self.size_bytes,
                    "label": self.label,
                    "confidence": self.confidence,
                    "stage_confidences": self.stage_confidences,
                    "pipeline_version": self.pipeline_version,
                    "processing_ms": self.processing_ms,
                    "warnings": self.warnings,
                    "errors": self.errors,
                }

        def _fake_classify(file_upload: UploadFile):
            return MockInternalClassificationResult(
                filename=file_upload.filename,
                label="unknown",
                confidence=0.0,
                mime_type=file_upload.content_type or "text/plain",
                size_bytes=(
                    len(file_upload.file.getbuffer())
                    if hasattr(file_upload.file, "getbuffer")
                    else 0
                ),
                pipeline_version=mock_settings.pipeline_version,
                processing_ms=10.0,
                warnings=[],
                errors=[],
            )

        mock_classify.side_effect = _fake_classify

        response = client.post(
            "/v1/files", files=_build_multipart_payload(3), headers=headers
        )

        assert response.status_code == 200, response.text
        assert response.headers.get("content-type", "").startswith("application/json")
        mock_validate.assert_called()
        assert mock_validate.call_count == 3

        payload = response.json()
        assert isinstance(payload, list)
        assert len(payload) == 3

        required_keys = {"filename", "label", "confidence", "request_id"}
        expected_request_id = headers["X-Request-ID"]

        for item in payload:
            assert isinstance(item, dict)
            assert required_keys.issubset(
                item.keys()
            ), f"Missing keys {required_keys - item.keys()} in {item}"
            assert isinstance(item["label"], str)
            assert 0.0 <= float(item["confidence"]) <= 1.0
            assert item["request_id"] == expected_request_id

        assert "X-Request-ID" in response.headers
        assert response.headers["X-Request-ID"] == expected_request_id


def test_upload_no_files(client: TestClient, headers: dict[str, str]) -> None:
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
    assert detail_item["type"] == "missing"
    assert detail_item["loc"] == ["body", "files"]
    assert "Field required" in detail_item["msg"]


def test_upload_empty_files_list(client: TestClient, headers: dict[str, str]) -> None:
    """
    Test sending a POST request where the 'files' field is present but empty.
    This should also result in a RequestValidationError caught by our custom handler.
    """
    # Sending `files=[]` for `List[UploadFile] = File(...)` will likely be
    # caught by FastAPI's validation as an invalid type or insufficient items
    # for the File field, resulting in a 422.
    response = client.post("/v1/files", files=[], headers=headers)

    assert response.status_code == 422
    payload = response.json()
    assert "error" in payload
    assert payload["error"]["code"] == "validation_error"
    assert payload["error"]["message"] == "Invalid request parameters."
    assert "details" in payload["error"]
    assert isinstance(payload["error"]["details"], list)
    # The exact detail message might vary, but it should indicate 'files' field is the issue
    assert any(
        "files" in detail.get("loc", []) for detail in payload["error"]["details"]
    ), "Error detail should reference the 'files' field."
    # Example messages: "Field required", "List should have at least 1 item after validation, not 0"
    assert any(
        "field required" in detail.get("msg", "").lower()
        or "ensure this value has at least 1 item" in detail.get("msg", "").lower()
        or "list should have at least 1 item" in detail.get("msg", "").lower()
        for detail in payload["error"]["details"]
    ), "Error message should indicate missing or insufficient files for the 'files' field."


def test_upload_exceeds_batch_limit(
    client: TestClient, headers: dict[str, str], mock_settings: MockSettings
) -> None:
    """Test uploading more files than the configured MAX_BATCH_SIZE."""
    mock_settings.max_batch_size = 2
    app.dependency_overrides[get_settings] = lambda: mock_settings

    files_payload = _build_multipart_payload(3)

    response = client.post("/v1/files", files=files_payload, headers=headers)

    assert response.status_code == 413  # This is an HTTPException raised by the route
    payload = response.json()
    # HTTPExceptions are handled by _http_exception_handler, which adds a top-level 'detail'
    assert payload["error"]["code"] == 413  # from the custom error payload
    assert payload["error"]["message"] == "Batch size 3 exceeds limit of 2."
    assert payload["detail"] == "Batch size 3 exceeds limit of 2."


def test_upload_with_validation_error(
    client: TestClient, headers: dict[str, str], mock_settings: MockSettings
) -> None:
    """Test the scenario where src.ingestion.validators.validate_file raises an HTTPException."""
    files_payload = _build_multipart_payload(2)

    validation_exception = HTTPException(
        status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
        detail="Unsupported file extension: .bad",
    )
    # Mock the validate_file function imported in src.api.routes.files
    with patch(
        "src.api.routes.files.validate_file", side_effect=validation_exception
    ) as mock_validate_file_in_route:
        response = client.post(
            "/v1/files", files=files_payload[0:1], headers=headers
        )  # Send one file that will trigger this

        assert response.status_code == 415
        payload = response.json()
        # This HTTPException is caught by _http_exception_handler
        assert payload["error"]["code"] == 415
        assert payload["error"]["message"] == "Unsupported file extension: .bad"
        assert payload["detail"] == "Unsupported file extension: .bad"
        mock_validate_file_in_route.assert_called_once()
