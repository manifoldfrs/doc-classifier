from __future__ import annotations

import uuid
from io import BytesIO
from typing import List
from unittest.mock import AsyncMock, patch, MagicMock

import pytest
import redis.asyncio as aioredis  # Import for type hint of mock_redis_client
from fastapi import HTTPException, UploadFile, status
from fastapi.testclient import TestClient

from src.api.app import app
from src.api.routes.jobs import get_redis_client  # Import for dependency override
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
def mock_redis_client_for_files_test() -> MagicMock:
    """Provides a MagicMock for Redis specifically for test_files.py client fixture."""
    mock_client = MagicMock(spec=aioredis.Redis)
    mock_client.ping = AsyncMock(return_value=True)
    mock_client.set = AsyncMock(return_value=True)  # For create_job
    # Add other necessary mock methods if create_job or run_job were to be deeply tested here.
    return mock_client


@pytest.fixture
def client(
    mock_settings: MockSettings, mock_redis_client_for_files_test: MagicMock
) -> TestClient:
    """Provides a TestClient instance with dependency overrides for settings and Redis."""
    app.dependency_overrides[get_settings] = lambda: mock_settings
    # The /v1/files route might trigger async job creation, which uses get_redis_client
    app.dependency_overrides[get_redis_client] = (
        lambda: mock_redis_client_for_files_test
    )

    # Ensure MockSettings used by this client has Redis attributes initialized
    if not hasattr(mock_settings, "redis_url") or mock_settings.redis_url is None:
        mock_settings.redis_host = getattr(mock_settings, "redis_host", "localhost")
        mock_settings.redis_port = getattr(mock_settings, "redis_port", 6379)
        mock_settings.redis_db = getattr(mock_settings, "redis_db", 0)
        mock_settings.redis_url = f"redis://{mock_settings.redis_host}:{mock_settings.redis_port}/{mock_settings.redis_db}"

    mock_settings.allowed_api_keys = ["test-api-key"]
    mock_settings.allowed_extensions = {"txt"}  # Keep it simple for these tests

    yield TestClient(app)
    app.dependency_overrides = {}  # Clean up overrides after test


@pytest.fixture
def headers(mock_settings: MockSettings) -> dict[str, str]:
    """Provides default headers including the API key and a test request ID."""
    if not mock_settings.allowed_api_keys:  # Should be set by client fixture
        mock_settings.allowed_api_keys = ["test-api-key-default-in-headers"]
    return {
        "x-api-key": mock_settings.allowed_api_keys[0],
        "X-Request-ID": f"test-req-id-{uuid.uuid4().hex}",
    }


def test_batch_upload_three_files_returns_expected_shape(
    client: TestClient, mock_settings: MockSettings, headers: dict[str, str]
) -> None:
    """Upload 3 files (sync path) and assert the 200-OK JSON structure."""
    with patch(
        "src.api.routes.files.classify", new_callable=AsyncMock
    ) as mock_classify:

        # Define a simple structure for the mocked classify's return value's .dict() method
        class StubClassificationResult:
            def __init__(
                self,
                filename,
                content_type,
                content_bytes,
                pipeline_version,
                request_id,
            ):
                self.filename = filename
                self.mime_type = content_type or "text/plain"
                self.size_bytes = len(content_bytes)
                self.label = "unknown"  # Default for mock
                self.confidence = 0.0  # Default for mock
                self.stage_confidences = {}
                self.pipeline_version = pipeline_version
                self.processing_ms = 10.0  # Dummy value
                self.warnings = []
                self.errors = []
                self.request_id = request_id  # This will be set by _classify_single

            def dict(self):  # Required by the schema creation
                return self.__dict__

        async def _fake_classify(file: UploadFile, request_id_from_caller: str):
            content_bytes = await file.read()
            await file.seek(0)
            # Note: The actual `classify` in `src.classification.pipeline` does not take request_id.
            # The `_classify_single` wrapper in `src.api.routes.files` passes it to the schema.
            # So, our _fake_classify here should mimic the return of the internal `classify`.
            internal_result_dict = StubClassificationResult(
                file.filename,
                file.content_type,
                content_bytes,
                mock_settings.pipeline_version,
                request_id_from_caller,
            ).dict()
            # Remove request_id as the internal classify function doesn't produce it.
            # It's added when constructing ClassificationResultSchema in _classify_single.
            del internal_result_dict["request_id"]
            return MagicMock(dict=lambda: internal_result_dict)

        # Patch the _classify_single helper to control its output directly for sync path
        with patch(
            "src.api.routes.files._classify_single", new_callable=AsyncMock
        ) as mock_classify_single_helper:

            async def fake_helper_output(file: UploadFile, batch_req_id: str):
                # This helper is what returns ClassificationResultSchema
                # The request_id in the schema should match the batch_req_id
                content_bytes = (
                    await file.read()
                )  # Reading again to simulate size if needed
                await file.seek(0)
                return MagicMock(
                    filename=file.filename,
                    mime_type=file.content_type or "text/plain",
                    size_bytes=len(content_bytes),
                    label="mocked_label",
                    confidence=0.55,
                    stage_confidences={},
                    pipeline_version=mock_settings.pipeline_version,
                    processing_ms=12.0,
                    request_id=batch_req_id,  # Crucial: use the passed batch_request_id
                    warnings=[],
                    errors=[],
                    model_dump=lambda by_alias: {  # Mock Pydantic V2's model_dump
                        "filename": file.filename,
                        "mime_type": file.content_type or "text/plain",
                        "size_bytes": len(content_bytes),
                        "label": "mocked_label",
                        "confidence": 0.55,
                        "stage_confidences": {},
                        "pipeline_version": mock_settings.pipeline_version,
                        "processing_ms": 12.0,
                        "request_id": batch_req_id,
                        "warnings": [],
                        "errors": [],
                    },
                )

            mock_classify_single_helper.side_effect = fake_helper_output

            response = client.post(
                "/v1/files", files=_build_multipart_payload(count=3), headers=headers
            )

            assert response.status_code == 200, response.text
            payload = response.json()
            assert isinstance(payload, list) and len(payload) == 3

            expected_request_id = headers["X-Request-ID"]
            assert response.headers.get("X-Request-ID") == expected_request_id

            for item in payload:
                assert item["request_id"] == expected_request_id
                assert item["label"] == "mocked_label"

            assert mock_classify_single_helper.call_count == 3
            # Check that _classify_single was called with the correct batch_request_id
            for call_args in mock_classify_single_helper.call_args_list:
                assert (
                    call_args[0][1] == expected_request_id
                )  # second arg to _classify_single


def test_upload_batch_exceeds_limit(
    client: TestClient, headers: dict[str, str], mock_settings: MockSettings
) -> None:
    mock_settings.max_batch_size = 2
    files_payload = _build_multipart_payload(count=3)
    response = client.post("/v1/files", files=files_payload, headers=headers)
    assert response.status_code == 413
    payload = response.json()
    assert payload["error"]["message"] == "Batch size 3 exceeds limit of 2."
    assert payload["error"]["request_id"] == headers["X-Request-ID"]


def test_upload_with_unsupported_extension(
    client: TestClient, headers: dict[str, str], mock_settings: MockSettings
) -> None:
    mock_settings.allowed_extensions = {"txt"}  # Ensure only txt is allowed
    files_payload = _build_multipart_payload(
        count=1, include_unsupported_ext=True
    )  # This adds a .zip
    response = client.post("/v1/files", files=files_payload, headers=headers)
    assert response.status_code == 415
    payload = response.json()
    assert "Unsupported file extension: .zip" in payload["error"]["message"]
    assert payload["error"]["request_id"] == headers["X-Request-ID"]


def test_upload_no_files_field_fastapi_validation(
    client: TestClient, headers: dict[str, str]
) -> None:
    response = client.post("/v1/files", data={"other_field": "value"}, headers=headers)
    assert response.status_code == 422
    payload = response.json()
    assert payload["error"]["code"] == "validation_error"
    assert "Field required" in payload["error"]["details"][0]["msg"]
    assert "files" in payload["error"]["details"][0]["loc"]
    assert payload["error"]["request_id"] == headers["X-Request-ID"]


def test_upload_empty_files_list(client: TestClient, headers: dict[str, str]) -> None:
    response = client.post(
        "/v1/files", files=[], headers=headers
    )  # Empty list for 'files'
    assert response.status_code == 422
    payload = response.json()
    assert payload["error"]["code"] == "validation_error"
    # Pydantic message for list min_items not met, or general "field required"
    assert any(
        "list should have at least 1 item" in detail["msg"].lower()
        or "field required" in detail["msg"].lower()
        for detail in payload["error"]["details"]
        if "files" in detail.get("loc", [])
    )
    assert payload["error"]["request_id"] == headers["X-Request-ID"]


def test_upload_with_validation_error_raised_by_route_validator(
    client: TestClient, headers: dict[str, str], mock_settings: MockSettings
) -> None:
    files_payload = _build_multipart_payload(count=1)  # One valid .txt file
    validation_exception = HTTPException(
        status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
        detail="Custom validator error: Deliberate fail.",
    )
    with patch("src.api.routes.files.validate_file", side_effect=validation_exception):
        response = client.post("/v1/files", files=files_payload, headers=headers)
        assert response.status_code == 415
        payload = response.json()
        assert payload["error"]["code"] == 415
        assert payload["error"]["message"] == "Custom validator error: Deliberate fail."
        assert payload["error"]["request_id"] == headers["X-Request-ID"]


@pytest.mark.asyncio
async def test_files_route_async_path_triggers_create_job(
    client: TestClient,
    headers: dict[str, str],
    mock_settings: MockSettings,
    mock_redis_client_for_files_test: MagicMock,
) -> None:
    """Test that an upload of > ASYNC_THRESHOLD files triggers async job creation."""
    # ASYNC_THRESHOLD is 10 in files.py
    num_files_for_async = 11
    files_payload = _build_multipart_payload(count=num_files_for_async)

    # Mock create_redis_job and run_redis_job from src.api.routes.files
    # because that's where they are imported and used.
    mocked_job_id = f"job_{uuid.uuid4().hex}"
    with (
        patch(
            "src.api.routes.files.create_redis_job",
            AsyncMock(return_value=mocked_job_id),
        ) as mock_create_job,
        patch("src.api.routes.files.run_redis_job", AsyncMock()) as mock_run_job,
        patch("asyncio.create_task") as mock_create_task,
    ):  # Also mock asyncio.create_task

        response = client.post("/v1/files", files=files_payload, headers=headers)

        assert response.status_code == 202  # Accepted for async processing
        payload = response.json()
        assert payload["job_id"] == mocked_job_id
        assert payload["status"] == "queued"
        assert response.headers["X-Request-ID"] == headers["X-Request-ID"]

        mock_create_job.assert_called_once()
        # Assert it was called with (num_files, redis_client_instance)
        assert mock_create_job.call_args[0][0] == num_files_for_async
        assert mock_create_job.call_args[0][1] is mock_redis_client_for_files_test

        # Assert that asyncio.create_task was called to run the job
        mock_create_task.assert_called_once()
        # The first argument to create_task is the coroutine, check its name or structure if complex
        # Here, we check if run_redis_job was the target of the task
        # The actual coroutine object passed to create_task is run_redis_job(...)
        # It's tricky to assert the exact coroutine object.
        # We can check if run_redis_job was prepared to be called.
        # The mock_run_job itself is what asyncio.create_task would eventually schedule.
        # Since run_redis_job is mocked, checking its call is more robust if create_task isn't failing.
        # However, the task is created, run_redis_job might not have been awaited yet by the test runner.
        # A better check: ensure create_task was called, and the args to run_redis_job are correct
        # via the args of the coroutine passed to create_task.
        # For simplicity, let's assume if create_task is called, run_job was its argument.

        # To check args for run_redis_job more deeply, you'd need to inspect mock_create_task.call_args[0][0]
        # which is the coroutine.
        # For now, ensuring it was called is a good step.
        # The `run_redis_job` itself is mocked so it won't actually run in this test.


def test_files_route_redis_connection_error_on_async(
    client: TestClient,
    headers: dict[str, str],
    mock_settings: MockSettings,
    mock_redis_client_for_files_test: MagicMock,
) -> None:
    """Test /v1/files async path when create_redis_job raises HTTPException due to Redis error."""
    num_files_for_async = 11
    files_payload = _build_multipart_payload(count=num_files_for_async)

    # Simulate create_redis_job failing due to Redis issue
    # It should raise an HTTPException(503)
    with patch(
        "src.api.routes.files.create_redis_job",
        AsyncMock(
            side_effect=HTTPException(
                status_code=503, detail="Redis connection failed for job creation"
            )
        ),
    ) as mock_create_job:

        response = client.post("/v1/files", files=files_payload, headers=headers)

        assert response.status_code == 503
        payload = response.json()
        assert "Redis connection failed for job creation" in payload["error"]["message"]
        assert payload["error"]["request_id"] == headers["X-Request-ID"]
        mock_create_job.assert_called_once()
