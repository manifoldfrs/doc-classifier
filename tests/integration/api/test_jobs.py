from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

from src.api.app import app
from src.api.routes.jobs import _JOB_REGISTRY, _REGISTRY_LOCK, JobStatus
from src.core.config import get_settings
from tests.conftest import MockSettings

pytestmark = [pytest.mark.integration]


@pytest.fixture
def client(mock_settings: MockSettings) -> TestClient:
    """Provides a TestClient instance with overridden settings for job tests."""
    app.dependency_overrides[get_settings] = lambda: mock_settings
    # Ensure a clean job registry for each test
    _JOB_REGISTRY.clear()
    return TestClient(app)


@pytest.mark.asyncio
async def test_get_job_not_found(
    client: TestClient, mock_settings: MockSettings
) -> None:
    """
    Tests GET /v1/jobs/{job_id} for a non-existent job ID.
    It should return 404 Not Found.
    """
    mock_settings.allowed_api_keys = ["test-key"]
    non_existent_job_id = uuid.uuid4().hex

    response = client.get(
        f"/v1/jobs/{non_existent_job_id}", headers={"x-api-key": "test-key"}
    )

    assert response.status_code == 404
    payload = response.json()
    assert payload["detail"] == f"Job '{non_existent_job_id}' not found."


@pytest.mark.asyncio
async def test_get_job_status_queued(
    client: TestClient, mock_settings: MockSettings
) -> None:
    """
    Tests GET /v1/jobs/{job_id} for a job that is still queued.
    """
    mock_settings.allowed_api_keys = ["test-key"]
    job_id = uuid.uuid4().hex

    # Manually add a job to the registry
    from src.api.routes.jobs import JobRecord  # Import here to avoid global issues

    async with _REGISTRY_LOCK:
        _JOB_REGISTRY[job_id] = JobRecord(total_files=5)
        _JOB_REGISTRY[job_id].status = JobStatus.queued

    response = client.get(f"/v1/jobs/{job_id}", headers={"x-api-key": "test-key"})

    assert response.status_code == 200
    payload = response.json()
    assert payload == {"job_id": job_id, "status": "queued"}


@pytest.mark.asyncio
async def test_get_job_status_processing(
    client: TestClient, mock_settings: MockSettings
) -> None:
    """
    Tests GET /v1/jobs/{job_id} for a job that is processing.
    """
    mock_settings.allowed_api_keys = ["test-key"]
    job_id = uuid.uuid4().hex
    from src.api.routes.jobs import JobRecord

    async with _REGISTRY_LOCK:
        _JOB_REGISTRY[job_id] = JobRecord(total_files=3)
        _JOB_REGISTRY[job_id].status = JobStatus.processing

    response = client.get(f"/v1/jobs/{job_id}", headers={"x-api-key": "test-key"})

    assert response.status_code == 200
    payload = response.json()
    assert payload == {"job_id": job_id, "status": "processing"}


@pytest.mark.asyncio
async def test_get_job_status_done_with_results(
    client: TestClient, mock_settings: MockSettings
) -> None:
    """
    Tests GET /v1/jobs/{job_id} for a job that is done, including results.
    """
    mock_settings.allowed_api_keys = ["test-key"]
    job_id = uuid.uuid4().hex
    from src.api.routes.jobs import JobRecord
    from src.api.schemas import ClassificationResultSchema  # Ensure schema is available

    # Create a mock result
    mock_result_data = {
        "filename": "test.pdf",
        "mime_type": "application/pdf",
        "size_bytes": 1024,
        "label": "invoice",
        "confidence": 0.95,
        "stage_confidences": {"filename": 0.9, "text": 0.95},
        "pipeline_version": mock_settings.pipeline_version,
        "processing_ms": 123.45,
        "request_id": uuid.uuid4().hex,
        "warnings": [],
        "errors": [],
    }
    # Ensure ClassificationResultSchema can be instantiated directly for the test
    # or that JobRecord.results correctly stores serializable data.
    # For this test, we'll assume ClassificationResultSchema can be created from a dict.

    # If ClassificationResultSchema directly inherits from ClassificationResult (dataclass)
    # and ClassificationResult is a dataclass, we can instantiate it.
    # If it's a Pydantic model, it should also work.
    try:
        # Attempt to create schema instance. This matches how it's done in jobs.py run_job
        # (via internal_result.dict() then ** unpacking)
        classification_result = ClassificationResultSchema(**mock_result_data)
    except Exception as e:
        pytest.fail(f"Failed to instantiate ClassificationResultSchema for test: {e}")

    async with _REGISTRY_LOCK:
        _JOB_REGISTRY[job_id] = JobRecord(total_files=1)
        _JOB_REGISTRY[job_id].status = JobStatus.done
        _JOB_REGISTRY[job_id].results = [classification_result]

    response = client.get(f"/v1/jobs/{job_id}", headers={"x-api-key": "test-key"})

    assert response.status_code == 200
    payload = response.json()

    assert payload["job_id"] == job_id
    assert payload["status"] == "done"
    assert len(payload["results"]) == 1
    # Pydantic's .dict() by default will convert UUIDs and other types to strings if needed.
    # We compare with mock_result_data which has string UUIDs.
    # The actual response schema should handle serialization correctly.
    # Here we'll check a few key fields.
    api_result = payload["results"][0]
    assert api_result["filename"] == mock_result_data["filename"]
    assert api_result["label"] == mock_result_data["label"]
    assert api_result["confidence"] == mock_result_data["confidence"]
    assert api_result["pipeline_version"] == mock_settings.pipeline_version
    assert api_result["request_id"] == mock_result_data["request_id"]
