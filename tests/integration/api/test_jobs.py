from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from starlette.datastructures import UploadFile

from src.api.app import app
from src.api.routes.jobs import (
    _JOB_REGISTRY,
    _REGISTRY_LOCK,
    JobRecord,
    JobStatus,
    create_job,
    run_job,
)
from src.api.schemas import ClassificationResultSchema
from src.core.config import get_settings
from tests.conftest import MockSettings

pytestmark = [pytest.mark.integration]


@pytest.fixture
def client(mock_settings: MockSettings) -> TestClient:
    """Provides a TestClient instance with overridden settings for job tests."""
    app.dependency_overrides[get_settings] = lambda: mock_settings
    mock_settings.allowed_api_keys = ["test-key"]
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture(autouse=True)
async def clear_job_registry_after_test():
    """Clears the in-memory job registry after each test."""
    try:
        yield
    finally:
        async with _REGISTRY_LOCK:
            _JOB_REGISTRY.clear()


async def _get_job_record(job_id: str) -> JobRecord | None:
    """Helper to safely get a job record from the registry."""
    async with _REGISTRY_LOCK:
        return _JOB_REGISTRY.get(job_id)


@pytest.mark.asyncio
async def test_create_job_successful(mock_settings: MockSettings) -> None:
    """Tests that create_job correctly initializes a job in the registry."""
    app.dependency_overrides[get_settings] = lambda: mock_settings
    total_files = 5
    job_id = await create_job(total_files)

    assert job_id is not None
    assert isinstance(job_id, str)

    record = await _get_job_record(job_id)
    assert record is not None
    assert record.status == JobStatus.queued
    assert record.total_files == total_files
    assert len(record.results) == 0


@pytest.mark.asyncio
async def test_run_job_successful_classification(mock_settings: MockSettings) -> None:
    """Tests run_job processing files successfully through a mocked classify."""
    app.dependency_overrides[get_settings] = lambda: mock_settings
    job_id = await create_job(total_files=2)

    raw_files_data = [
        ("file1.txt", "text/plain", b"content1"),
        ("file2.pdf", "application/pdf", b"content2"),
    ]

    mock_dict_output1 = {
        "filename": "file1.txt",
        "mime_type": "text/plain",
        "size_bytes": 8,
        "label": "text_doc",
        "confidence": 0.9,
        "stage_confidences": {},
        "pipeline_version": mock_settings.pipeline_version,
        "processing_ms": 10.0,
        "warnings": [],
        "errors": [],
    }
    mock_dict_output2 = {
        "filename": "file2.pdf",
        "mime_type": "application/pdf",
        "size_bytes": 8,
        "label": "invoice",
        "confidence": 0.95,
        "stage_confidences": {},
        "pipeline_version": mock_settings.pipeline_version,
        "processing_ms": 20.0,
        "warnings": [],
        "errors": [],
    }

    mock_internal_classify_result1 = MagicMock()
    mock_internal_classify_result1.dict.return_value = mock_dict_output1

    mock_internal_classify_result2 = MagicMock()
    mock_internal_classify_result2.dict.return_value = mock_dict_output2

    mock_classify_fn = AsyncMock(
        side_effect=[mock_internal_classify_result1, mock_internal_classify_result2]
    )

    with patch("src.api.routes.jobs.classify", mock_classify_fn):
        await run_job(job_id, raw_files_data)

    record = await _get_job_record(job_id)
    assert record is not None
    assert record.status == JobStatus.done
    assert len(record.results) == 2

    assert isinstance(record.results[0], ClassificationResultSchema)
    assert record.results[0].filename == "file1.txt"
    assert record.results[0].label == "text_doc"
    assert record.results[0].request_id is not None
    assert isinstance(record.results[0].request_id, str)

    assert isinstance(record.results[1], ClassificationResultSchema)
    assert record.results[1].filename == "file2.pdf"
    assert record.results[1].label == "invoice"
    assert record.results[1].request_id is not None
    assert isinstance(record.results[1].request_id, str)

    assert mock_classify_fn.call_count == 2
    first_call_args = mock_classify_fn.call_args_list[0][0]
    assert isinstance(first_call_args[0], UploadFile)
    assert first_call_args[0].filename == "file1.txt"


@pytest.mark.asyncio
async def test_run_job_classification_error_handling(
    mock_settings: MockSettings,
) -> None:
    """Tests run_job handling an error during one of the classifications."""
    app.dependency_overrides[get_settings] = lambda: mock_settings
    job_id = await create_job(total_files=2)
    raw_files_data = [
        ("error_file.txt", "text/plain", b"error_content"),
        ("good_file.txt", "text/plain", b"good_content"),
    ]

    mock_dict_output_good = {
        "filename": "good_file.txt",
        "mime_type": "text/plain",
        "size_bytes": 12,
        "label": "text_doc",
        "confidence": 0.8,
        "stage_confidences": {},
        "pipeline_version": mock_settings.pipeline_version,
        "processing_ms": 15.0,
        "warnings": [],
        "errors": [],
    }
    mock_internal_classify_result_good = MagicMock()
    mock_internal_classify_result_good.dict.return_value = mock_dict_output_good

    mock_classify_fn = AsyncMock(
        side_effect=[
            RuntimeError("Simulated classification failure"),
            mock_internal_classify_result_good,
        ]
    )

    with patch("src.api.routes.jobs.classify", mock_classify_fn):
        await run_job(job_id, raw_files_data)

    record = await _get_job_record(job_id)
    assert record is not None
    assert record.status == JobStatus.done
    assert len(record.results) == 2

    assert record.results[0].filename == "error_file.txt"
    assert record.results[0].label == "error"
    assert record.results[0].confidence == 0.0
    assert len(record.results[0].errors) == 1
    assert record.results[0].errors[0]["code"] == "classification_error"
    assert "Simulated classification failure" in record.results[0].errors[0]["message"]
    assert record.results[0].request_id is not None

    assert record.results[1].filename == "good_file.txt"
    assert record.results[1].label == "text_doc"
    assert record.results[1].request_id is not None


@pytest.mark.asyncio
async def test_run_job_job_not_found(mock_settings: MockSettings) -> None:
    """Tests run_job when the job_id does not exist in the registry."""
    app.dependency_overrides[get_settings] = lambda: mock_settings
    non_existent_job_id = uuid.uuid4().hex
    raw_files_data = [("file.txt", "text/plain", b"content")]

    original_registry_size = len(_JOB_REGISTRY)

    await run_job(non_existent_job_id, raw_files_data)

    assert len(_JOB_REGISTRY) == original_registry_size
    record = await _get_job_record(non_existent_job_id)
    assert record is None


@pytest.mark.asyncio
async def test_get_job_not_found(
    client: TestClient, mock_settings: MockSettings
) -> None:
    """
    Tests GET /v1/jobs/{job_id} for a non-existent job ID.
    It should return 404 Not Found.
    """
    non_existent_job_id = uuid.uuid4().hex

    response = client.get(
        f"/v1/jobs/{non_existent_job_id}", headers={"x-api-key": "test-key"}
    )

    assert response.status_code == 404
    payload = response.json()
    assert payload["error"]["code"] == 404
    assert payload["error"]["message"] == f"Job '{non_existent_job_id}' not found."
    assert payload["detail"] == f"Job '{non_existent_job_id}' not found."


@pytest.mark.asyncio
async def test_get_job_status_queued(
    client: TestClient, mock_settings: MockSettings
) -> None:
    """
    Tests GET /v1/jobs/{job_id} for a job that is still queued.
    """
    job_id = await create_job(total_files=5)

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
    job_id = await create_job(total_files=3)
    record = await _get_job_record(job_id)
    if record:
        async with _REGISTRY_LOCK:
            record.status = JobStatus.processing

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
    job_id = await create_job(total_files=1)

    mock_schema_data = {
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
    classification_result_schema = ClassificationResultSchema(**mock_schema_data)

    record = await _get_job_record(job_id)
    if record:
        async with _REGISTRY_LOCK:
            record.status = JobStatus.done
            record.results = [classification_result_schema]

    response = client.get(f"/v1/jobs/{job_id}", headers={"x-api-key": "test-key"})

    assert response.status_code == 200
    payload = response.json()

    assert payload["job_id"] == job_id
    assert payload["status"] == "done"
    assert len(payload["results"]) == 1

    api_result = payload["results"][0]
    assert api_result["filename"] == mock_schema_data["filename"]
    assert api_result["label"] == mock_schema_data["label"]
    assert api_result["confidence"] == mock_schema_data["confidence"]
    assert api_result["pipeline_version"] == mock_settings.pipeline_version
    assert api_result["request_id"] == mock_schema_data["request_id"]
