from __future__ import annotations

import json
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import redis.asyncio as aioredis
from fastapi import HTTPException
from fastapi.testclient import TestClient
from starlette.datastructures import UploadFile

# Import specific exceptions from redis.exceptions for robust error simulation
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import TimeoutError as RedisTimeoutError

from src.api.app import app
from src.api.routes.jobs import (
    _JOB_KEY_PREFIX,
    JobRecord,
    JobStatus,
    create_job,
    get_redis_client,
    run_job,
    close_redis_client,
)
from src.api.schemas import ClassificationResultSchema
from src.core.config import Settings, get_settings
from tests.conftest import MockSettings

pytestmark = [pytest.mark.integration]


@pytest.fixture
def mock_settings_for_jobs(mock_settings: MockSettings) -> MockSettings:
    """Specific settings for job tests, e.g., Redis config if needed."""
    mock_settings.allowed_api_keys = ["test-key"]
    mock_settings.redis_host = "localhost"
    mock_settings.redis_port = 6379
    mock_settings.redis_db = 0
    # Construct redis_url if not already set by MockSettings.__init__ based on other values
    if not mock_settings.redis_url:
        mock_settings.redis_url = f"redis://{mock_settings.redis_host}:{mock_settings.redis_port}/{mock_settings.redis_db}"
    return mock_settings


@pytest.fixture
async def mock_redis_client(
    mock_settings_for_jobs: MockSettings,
) -> aioredis.Redis:
    """
    Provides a MagicMock for the Redis client.
    This allows testing without a live Redis instance.
    """
    mock_client = MagicMock(spec=aioredis.Redis)
    mock_client.ping = AsyncMock(return_value=True)
    mock_client.get = AsyncMock(return_value=None)
    mock_client.set = AsyncMock(return_value=True)
    mock_client.delete = AsyncMock(return_value=1)
    mock_client.keys = AsyncMock(return_value=[])
    mock_client.close = AsyncMock(return_value=None)
    return mock_client


@pytest.fixture
def client(
    mock_settings_for_jobs: MockSettings, mock_redis_client: aioredis.Redis
) -> TestClient:
    """Provides a TestClient instance with overridden settings and Redis client for job tests."""
    app.dependency_overrides[get_settings] = lambda: mock_settings_for_jobs
    app.dependency_overrides[get_redis_client] = lambda: mock_redis_client
    yield TestClient(app, raise_server_exceptions=False)
    app.dependency_overrides = {}


@pytest.fixture(autouse=True)
async def clear_redis_after_test(mock_redis_client: aioredis.Redis):
    try:
        yield
    finally:
        # Reset call counts for a clean state in each test that uses the mock_redis_client
        if hasattr(mock_redis_client, "reset_mock"):  # For MagicMock
            mock_redis_client.reset_mock()
        else:  # For AsyncMock directly on functions if that were the case
            for method_name in ["ping", "get", "set", "delete", "keys", "close"]:
                if hasattr(getattr(mock_redis_client, method_name), "reset_mock"):
                    getattr(mock_redis_client, method_name).reset_mock()


async def _get_job_record_from_redis(
    job_id: str, redis_client: aioredis.Redis
) -> JobRecord | None:
    job_key = f"{_JOB_KEY_PREFIX}{job_id}"
    job_data_json_bytes = await redis_client.get(job_key)
    if job_data_json_bytes:
        return JobRecord.model_validate_json(job_data_json_bytes.decode("utf-8"))
    return None


@pytest.mark.asyncio
async def test_create_job_successful(
    mock_settings_for_jobs: MockSettings, mock_redis_client: aioredis.Redis
) -> None:
    total_files = 5
    job_id = await create_job(total_files, mock_redis_client)

    assert job_id is not None
    assert isinstance(job_id, str)

    expected_job_key = f"{_JOB_KEY_PREFIX}{job_id}"
    mock_redis_client.set.assert_called_once()
    call_args = mock_redis_client.set.call_args
    assert call_args[0][0] == expected_job_key

    job_data_json_str = call_args[0][1]
    job_data_from_redis_call = json.loads(job_data_json_str)

    assert job_data_from_redis_call["status"] == JobStatus.queued.value
    assert job_data_from_redis_call["total_files"] == total_files
    assert len(job_data_from_redis_call["results"]) == 0
    assert job_data_from_redis_call["job_id"] == job_id
    assert "ex" in call_args[1] and call_args[1]["ex"] is not None


@pytest.mark.asyncio
async def test_run_job_successful_classification(
    mock_settings_for_jobs: MockSettings, mock_redis_client: aioredis.Redis
) -> None:
    total_files = 2
    job_id = uuid.uuid4().hex
    job_key = f"{_JOB_KEY_PREFIX}{job_id}"

    initial_job_record = JobRecord(
        job_id=job_id, total_files=total_files, status=JobStatus.queued
    )
    mock_redis_client.get.return_value = initial_job_record.model_dump_json().encode(
        "utf-8"
    )

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
        "pipeline_version": mock_settings_for_jobs.pipeline_version,
        "processing_ms": 10.0,
        "warnings": [],
        "errors": [],
        "request_id": uuid.uuid4().hex,
    }
    mock_dict_output2 = {
        "filename": "file2.pdf",
        "mime_type": "application/pdf",
        "size_bytes": 8,
        "label": "invoice",
        "confidence": 0.95,
        "stage_confidences": {},
        "pipeline_version": mock_settings_for_jobs.pipeline_version,
        "processing_ms": 20.0,
        "warnings": [],
        "errors": [],
        "request_id": uuid.uuid4().hex,
    }

    mock_internal_classify_result1 = MagicMock()
    mock_internal_classify_result1.dict.return_value = mock_dict_output1
    mock_internal_classify_result2 = MagicMock()
    mock_internal_classify_result2.dict.return_value = mock_dict_output2

    mock_classify_fn = AsyncMock(
        side_effect=[mock_internal_classify_result1, mock_internal_classify_result2]
    )

    with patch("src.api.routes.jobs.classify", mock_classify_fn):
        await run_job(job_id, raw_files_data, mock_redis_client, mock_settings_for_jobs)

    assert mock_redis_client.set.call_count == 2
    final_set_call_args = mock_redis_client.set.call_args_list[-1]
    assert final_set_call_args[0][0] == job_key

    final_job_data_json_str = final_set_call_args[0][1]
    final_job_record = JobRecord.model_validate_json(final_job_data_json_str)

    assert final_job_record.status == JobStatus.done
    assert len(final_job_record.results) == 2
    assert final_job_record.results[0].filename == "file1.txt"
    assert final_job_record.results[1].filename == "file2.pdf"
    assert mock_classify_fn.call_count == 2


@pytest.mark.asyncio
async def test_run_job_classification_error_handling(
    mock_settings_for_jobs: MockSettings,
    mock_redis_client: aioredis.Redis,
) -> None:
    total_files = 2
    job_id = uuid.uuid4().hex
    initial_job_record = JobRecord(
        job_id=job_id, total_files=total_files, status=JobStatus.queued
    )
    mock_redis_client.get.return_value = initial_job_record.model_dump_json().encode(
        "utf-8"
    )

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
        "pipeline_version": mock_settings_for_jobs.pipeline_version,
        "processing_ms": 15.0,
        "warnings": [],
        "errors": [],
        "request_id": uuid.uuid4().hex,
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
        await run_job(job_id, raw_files_data, mock_redis_client, mock_settings_for_jobs)

    final_set_call_args = mock_redis_client.set.call_args_list[-1]
    final_job_record = JobRecord.model_validate_json(final_set_call_args[0][1])

    assert final_job_record.status == JobStatus.done
    assert len(final_job_record.results) == 2
    assert final_job_record.results[0].filename == "error_file.txt"
    assert final_job_record.results[0].label == "error"
    assert (
        "Simulated classification failure"
        in final_job_record.results[0].errors[0]["message"]
    )
    assert final_job_record.results[1].filename == "good_file.txt"


@pytest.mark.asyncio
async def test_run_job_job_not_found_in_redis(
    mock_settings_for_jobs: MockSettings,
    mock_redis_client: aioredis.Redis,
) -> None:
    non_existent_job_id = uuid.uuid4().hex
    raw_files_data = [("file.txt", "text/plain", b"content")]
    mock_redis_client.get.return_value = None

    with patch(
        "src.api.routes.jobs.classify", new_callable=AsyncMock
    ) as mock_classify_fn:
        await run_job(
            non_existent_job_id,
            raw_files_data,
            mock_redis_client,
            mock_settings_for_jobs,
        )
        mock_classify_fn.assert_not_called()
    mock_redis_client.set.assert_not_called()


@pytest.mark.asyncio
async def test_get_job_not_found(
    client: TestClient, mock_redis_client: aioredis.Redis
) -> None:
    non_existent_job_id = uuid.uuid4().hex
    mock_redis_client.get.return_value = None

    response = client.get(
        f"/v1/jobs/{non_existent_job_id}", headers={"x-api-key": "test-key"}
    )
    assert response.status_code == 404
    assert response.json()["detail"] == f"Job '{non_existent_job_id}' not found."


@pytest.mark.asyncio
async def test_get_job_status_queued(
    client: TestClient, mock_redis_client: aioredis.Redis
) -> None:
    job_id = uuid.uuid4().hex
    queued_job_record = JobRecord(job_id=job_id, total_files=5, status=JobStatus.queued)
    mock_redis_client.get.return_value = queued_job_record.model_dump_json().encode(
        "utf-8"
    )

    response = client.get(f"/v1/jobs/{job_id}", headers={"x-api-key": "test-key"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["job_id"] == job_id
    assert payload["status"] == JobStatus.queued.value


@pytest.mark.asyncio
async def test_get_job_status_processing(
    client: TestClient, mock_redis_client: aioredis.Redis
) -> None:
    job_id = uuid.uuid4().hex
    processing_job_record = JobRecord(
        job_id=job_id, total_files=3, status=JobStatus.processing
    )
    mock_redis_client.get.return_value = processing_job_record.model_dump_json().encode(
        "utf-8"
    )

    response = client.get(f"/v1/jobs/{job_id}", headers={"x-api-key": "test-key"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["job_id"] == job_id
    assert payload["status"] == JobStatus.processing.value


@pytest.mark.asyncio
async def test_get_job_status_done_with_results(
    client: TestClient,
    mock_redis_client: aioredis.Redis,
    mock_settings_for_jobs: MockSettings,
) -> None:
    job_id = uuid.uuid4().hex
    result_request_id = uuid.uuid4().hex
    mock_schema_data = {
        "filename": "test.pdf",
        "mime_type": "application/pdf",
        "size_bytes": 1024,
        "label": "invoice",
        "confidence": 0.95,
        "stage_confidences": {"filename": 0.9, "text": 0.95},
        "pipeline_version": mock_settings_for_jobs.pipeline_version,
        "processing_ms": 123.45,
        "request_id": result_request_id,
        "warnings": [],
        "errors": [],
    }
    classification_result_schema = ClassificationResultSchema(**mock_schema_data)
    done_job_record = JobRecord(
        job_id=job_id,
        total_files=1,
        status=JobStatus.done,
        results=[classification_result_schema],
    )
    mock_redis_client.get.return_value = done_job_record.model_dump_json().encode(
        "utf-8"
    )

    response = client.get(f"/v1/jobs/{job_id}", headers={"x-api-key": "test-key"})
    assert response.status_code == 200
    payload = response.json()

    assert payload["job_id"] == job_id
    assert payload["status"] == JobStatus.done.value
    assert len(payload["results"]) == 1
    api_result = payload["results"][0]
    assert api_result["filename"] == mock_schema_data["filename"]
    assert api_result["request_id"] == result_request_id


@pytest.mark.asyncio
async def test_run_job_loop_general_exception(
    mock_settings_for_jobs: MockSettings,
    mock_redis_client: aioredis.Redis,
) -> None:
    total_files = 1
    job_id = uuid.uuid4().hex
    initial_job_record = JobRecord(
        job_id=job_id, total_files=total_files, status=JobStatus.queued
    )
    mock_redis_client.get.return_value = initial_job_record.model_dump_json().encode(
        "utf-8"
    )

    raw_files_data = [("file1.txt", "text/plain", b"content1")]

    with patch(
        "src.api.routes.jobs._build_upload_from_bytes",
        side_effect=ValueError("Bad file data during build"),
    ):
        await run_job(job_id, raw_files_data, mock_redis_client, mock_settings_for_jobs)

    assert mock_redis_client.set.call_count == 2
    final_set_call_args = mock_redis_client.set.call_args_list[1]
    final_job_record = JobRecord.model_validate_json(final_set_call_args[0][1])

    assert final_job_record.status == JobStatus.failed
    assert (
        "Job processing loop failed: Bad file data during build"
        in final_job_record.error_message
    )
    assert len(final_job_record.results) == 0


@pytest.mark.asyncio
async def test_get_job_redis_connection_error(
    client: TestClient, mock_redis_client: aioredis.Redis
):
    job_id = uuid.uuid4().hex
    mock_redis_client.get.side_effect = RedisConnectionError("Redis unavailable")

    response = client.get(f"/v1/jobs/{job_id}", headers={"x-api-key": "test-key"})

    assert response.status_code == 503
    payload = response.json()
    assert "Failed to retrieve job from Redis." in payload["detail"]


@pytest.mark.asyncio
async def test_create_job_redis_connection_error(
    mock_settings_for_jobs: MockSettings, mock_redis_client: aioredis.Redis
):
    """Test create_job when Redis connection fails during set."""
    total_files = 1
    mock_redis_client.set.side_effect = RedisConnectionError(
        "Cannot connect to Redis for set"
    )

    with pytest.raises(HTTPException) as exc_info:
        await create_job(total_files, mock_redis_client)

    assert exc_info.value.status_code == 503
    assert "Failed to create job in Redis." in exc_info.value.detail


@pytest.mark.asyncio
async def test_run_job_redis_get_connection_error(
    mock_settings_for_jobs: MockSettings, mock_redis_client: aioredis.Redis
):
    """Test run_job when initial get from Redis fails."""
    job_id = uuid.uuid4().hex
    raw_files_data = [("file1.txt", "text/plain", b"content1")]
    mock_redis_client.get.side_effect = RedisConnectionError(
        "Failed to get job from Redis"
    )
    initial_set_call_count = mock_redis_client.set.call_count

    await run_job(job_id, raw_files_data, mock_redis_client, mock_settings_for_jobs)
    assert mock_redis_client.set.call_count == initial_set_call_count


@pytest.mark.asyncio
async def test_get_job_deserialization_error(
    client: TestClient, mock_redis_client: aioredis.Redis
):
    job_id = uuid.uuid4().hex
    mock_redis_client.get.return_value = json.dumps(
        {"job_id": job_id, "status": "done", "results": "this_is_not_a_list_of_schemas"}
    ).encode("utf-8")

    response = client.get(f"/v1/jobs/{job_id}", headers={"x-api-key": "test-key"})

    assert response.status_code == 500
    payload = response.json()
    assert f"Failed to process job data for job '{job_id}'." in payload["detail"]


@pytest.mark.asyncio
async def test_run_job_redis_set_processing_status_fails(
    mock_settings_for_jobs: MockSettings, mock_redis_client: aioredis.Redis
):
    """Test run_job when setting status to 'processing' in Redis fails but final set succeeds."""
    total_files = 1
    job_id = uuid.uuid4().hex
    initial_job_record = JobRecord(
        job_id=job_id, total_files=total_files, status=JobStatus.queued
    )
    mock_redis_client.get.return_value = initial_job_record.model_dump_json().encode(
        "utf-8"
    )

    mock_redis_client.set.side_effect = [
        RedisConnectionError("Failed to set processing status"),
        AsyncMock(return_value=True),
    ]

    raw_files_data = [("file1.txt", "text/plain", b"content1")]
    mock_classify_result_dict = {
        "filename": "file1.txt",
        "mime_type": "text/plain",
        "size_bytes": 8,
        "label": "text_doc",
        "confidence": 0.9,
        "stage_confidences": {},
        "pipeline_version": mock_settings_for_jobs.pipeline_version,
        "processing_ms": 10.0,
        "warnings": [],
        "errors": [],
        "request_id": uuid.uuid4().hex,
    }
    mock_internal_classify_result = MagicMock()
    mock_internal_classify_result.dict.return_value = mock_classify_result_dict
    mock_classify_fn = AsyncMock(return_value=mock_internal_classify_result)

    with (
        patch("src.api.routes.jobs.classify", mock_classify_fn),
        patch("src.api.routes.jobs.logger.error") as mock_logger_error,
    ):
        await run_job(job_id, raw_files_data, mock_redis_client, mock_settings_for_jobs)

    mock_logger_error.assert_any_call(
        "redis_set_processing_failed",
        job_id=job_id,
        error="Failed to set processing status",
    )
    assert mock_redis_client.set.call_count == 2

    final_set_call_args = mock_redis_client.set.call_args_list[1]
    final_job_data_json_str = final_set_call_args[0][1]
    final_job_record = JobRecord.model_validate_json(final_job_data_json_str)

    assert final_job_record.status == JobStatus.done
    assert len(final_job_record.results) == 1
    assert final_job_record.results[0].filename == "file1.txt"
    mock_classify_fn.assert_called_once()


@pytest.mark.asyncio
async def test_run_job_redis_set_final_status_fails(
    mock_settings_for_jobs: MockSettings, mock_redis_client: aioredis.Redis
):
    """Test run_job when setting the final status (done/failed) in Redis fails."""
    total_files = 1
    job_id = uuid.uuid4().hex
    initial_job_record = JobRecord(
        job_id=job_id, total_files=total_files, status=JobStatus.queued
    )
    mock_redis_client.get.return_value = initial_job_record.model_dump_json().encode(
        "utf-8"
    )

    # First 'set' (to processing) succeeds, second 'set' (to done) fails.
    mock_redis_client.set.side_effect = [
        AsyncMock(return_value=True),  # For processing status update
        RedisConnectionError(
            "Failed to set final job status"
        ),  # For final status update
    ]

    raw_files_data = [("file1.txt", "text/plain", b"content1")]
    mock_classify_result_dict = {
        "filename": "file1.txt",
        "mime_type": "text/plain",
        "size_bytes": 8,
        "label": "text_doc",
        "confidence": 0.9,
        "stage_confidences": {},
        "pipeline_version": mock_settings_for_jobs.pipeline_version,
        "processing_ms": 10.0,
        "warnings": [],
        "errors": [],
        "request_id": uuid.uuid4().hex,
    }
    mock_internal_classify_result = MagicMock()
    mock_internal_classify_result.dict.return_value = mock_classify_result_dict
    mock_classify_fn = AsyncMock(return_value=mock_internal_classify_result)

    with (
        patch("src.api.routes.jobs.classify", mock_classify_fn),
        patch("src.api.routes.jobs.logger.error") as mock_logger_error,
    ):
        await run_job(job_id, raw_files_data, mock_redis_client, mock_settings_for_jobs)

    # Check that logger.error was called for the Redis 'set final status' failure
    mock_logger_error.assert_any_call(
        "redis_update_job_failed",
        job_id=job_id,
        error="Failed to set final job status",
        exc_info=True,
    )

    # run_job attempts two 'set' calls:
    # 1. To mark as 'processing' (succeeds)
    # 2. To mark as 'done' (fails)
    assert mock_redis_client.set.call_count == 2
    mock_classify_fn.assert_called_once()  # Ensure classification still happened

    # Optional: check the arguments of the failed set call if necessary
    # calls = mock_redis_client.set.call_args_list
    # assert calls[0][0][1] contains '"status": "processing"'
    # assert calls[1][0][1] contains '"status": "done"'


@pytest.mark.asyncio
async def test_get_redis_client_initial_ping_fails(
    mock_settings_for_jobs: MockSettings,  # Ensure settings are available
):
    """Test get_redis_client when the initial Redis ping fails."""
    original_overrides = app.dependency_overrides.copy()
    app.dependency_overrides.clear()
    app.dependency_overrides[get_settings] = lambda: mock_settings_for_jobs
    # Do not override get_redis_client here; we want the actual function to run with patched internals.

    try:
        # Ensure _REDIS_CLIENT is None so initialization is attempted
        with patch("src.api.routes.jobs._REDIS_CLIENT", None):
            # Mock aioredis.from_url to return a client that fails on ping
            mock_failing_redis_instance = AsyncMock(spec=aioredis.Redis)
            mock_failing_redis_instance.ping = AsyncMock(
                side_effect=RedisConnectionError("Ping failed")
            )

            with patch(
                "redis.asyncio.from_url", return_value=mock_failing_redis_instance
            ) as mock_from_url:
                local_test_client = TestClient(app, raise_server_exceptions=False)

                job_id_for_test = uuid.uuid4().hex
                response = local_test_client.get(
                    f"/v1/jobs/{job_id_for_test}", headers={"x-api-key": "test-key"}
                )

                assert response.status_code == 503
                assert "Could not connect to Redis" in response.json()["detail"]
                mock_from_url.assert_called_once_with(mock_settings_for_jobs.redis_url)
                mock_failing_redis_instance.ping.assert_called_once()
    finally:
        app.dependency_overrides = original_overrides


@pytest.mark.asyncio
async def test_get_redis_client_already_none(
    mock_settings_for_jobs: MockSettings,  # Add mock_settings_for_jobs
):
    """Test get_redis_client when _REDIS_CLIENT is already None from a previous failed attempt."""
    original_overrides = app.dependency_overrides.copy()
    app.dependency_overrides.clear()
    app.dependency_overrides[get_settings] = lambda: mock_settings_for_jobs
    # Do not override get_redis_client here.

    try:
        # Patch the global _REDIS_CLIENT in the jobs module to be None.
        # The actual get_redis_client is expected to see this and raise "Redis client not available"
        # without attempting a connection, if it's coded for that specific path.
        with patch("src.api.routes.jobs._REDIS_CLIENT", None):
            local_test_client = TestClient(app, raise_server_exceptions=False)

            job_id_for_test = uuid.uuid4().hex
            response = local_test_client.get(
                f"/v1/jobs/{job_id_for_test}", headers={"x-api-key": "test-key"}
            )

            assert response.status_code == 503
            assert "Could not connect to Redis" in response.json()["detail"]
            # No check for from_url or ping, as the premise is that connection is not attempted.
    finally:
        app.dependency_overrides = original_overrides


@pytest.mark.asyncio
async def test_run_job_file_processing_unexpected_generic_exception(
    mock_settings_for_jobs: MockSettings, mock_redis_client: aioredis.Redis
):
    """Test run_job when a generic Exception (non-StageExecutionError) occurs during classify for one file."""
    total_files = 2
    job_id = uuid.uuid4().hex
    initial_job_record = JobRecord(
        job_id=job_id, total_files=total_files, status=JobStatus.queued
    )
    mock_redis_client.get.return_value = initial_job_record.model_dump_json().encode(
        "utf-8"
    )

    raw_files_data = [
        ("generic_error_file.txt", "text/plain", b"error_content"),
        ("good_file.pdf", "application/pdf", b"good_content"),
    ]

    mock_good_file_dict = {
        "filename": "good_file.pdf",
        "mime_type": "application/pdf",
        "size_bytes": 12,
        "label": "invoice",
        "confidence": 0.95,
        "stage_confidences": {},
        "pipeline_version": mock_settings_for_jobs.pipeline_version,
        "processing_ms": 20.0,
        "warnings": [],
        "errors": [],
        "request_id": uuid.uuid4().hex,
    }
    mock_internal_good_result = MagicMock()
    mock_internal_good_result.dict.return_value = mock_good_file_dict

    # Simulate classify raising a generic Exception for the first file, and succeeding for the second
    mock_classify_fn = AsyncMock(
        side_effect=[
            TypeError("A very unexpected type error during classification"),
            mock_internal_good_result,
        ]
    )

    with (
        patch("src.api.routes.jobs.classify", mock_classify_fn),
        patch("src.api.routes.jobs.logger.error") as mock_logger_error,
    ):
        await run_job(job_id, raw_files_data, mock_redis_client, mock_settings_for_jobs)

    # Check that the specific logger for unexpected errors was called
    mock_logger_error.assert_any_call(
        "job_file_classification_unexpected_error",
        job_id=job_id,
        filename="generic_error_file.txt",
        error="A very unexpected type error during classification",
        exc_info=True,
    )

    # Check the final job state
    final_set_call_args = mock_redis_client.set.call_args_list[
        -1
    ]  # Second set call (final status)
    final_job_record = JobRecord.model_validate_json(final_set_call_args[0][1])

    assert (
        final_job_record.status == JobStatus.done
    )  # Job completes even if one file has generic error
    assert len(final_job_record.results) == 2

    # Check the result for the file that had a generic error
    error_file_result = next(
        r for r in final_job_record.results if r.filename == "generic_error_file.txt"
    )
    assert error_file_result.label == "error"
    assert error_file_result.confidence == 0.0
    assert len(error_file_result.errors) == 1
    assert error_file_result.errors[0]["code"] == "classification_error"
    assert (
        "A very unexpected type error during classification"
        in error_file_result.errors[0]["message"]
    )

    # Check the result for the file that processed successfully
    good_file_result = next(
        r for r in final_job_record.results if r.filename == "good_file.pdf"
    )
    assert good_file_result.label == "invoice"
    assert good_file_result.confidence == 0.95


@pytest.mark.asyncio
async def test_close_redis_client_when_initialized(
    mock_redis_client: aioredis.Redis,  # Use the standard mock_redis_client fixture
    client: TestClient,  # To trigger get_redis_client via an endpoint
    mock_settings_for_jobs: MockSettings,  # To ensure settings are available
):
    """Test close_redis_client when _REDIS_CLIENT has been initialized."""
    # Step 1: Ensure _REDIS_CLIENT is initialized by making a call through the TestClient
    # The client fixture already overrides get_redis_client to return mock_redis_client.
    # So, any call using this client will effectively "initialize" _REDIS_CLIENT to mock_redis_client
    # for the purpose of how get_redis_client normally works with the global.

    # To properly test close_redis_client, which acts on the *actual* global _REDIS_CLIENT
    # in src.api.routes.jobs, we need to ensure that global variable is set.
    # The easiest way is to patch it to be our mock_redis_client for the duration of this test.

    with (
        patch("src.api.routes.jobs._REDIS_CLIENT", mock_redis_client),
        patch("src.api.routes.jobs.logger.info") as mock_logger_info,
    ):

        # Verify _REDIS_CLIENT is indeed our mock before closing
        # This import is safe here as we are testing the module's state
        from src.api.routes import jobs as jobs_module

        assert jobs_module._REDIS_CLIENT == mock_redis_client

        await close_redis_client()  # Call the function we are testing

        # Verify that the mock_redis_client's close method was called
        mock_redis_client.close.assert_called_once()

        # Verify that the global _REDIS_CLIENT in the module is set to None
        assert jobs_module._REDIS_CLIENT is None

        # Verify logger call
        mock_logger_info.assert_any_call("redis_client_closed")


@pytest.mark.asyncio
async def test_close_redis_client_when_already_none(
    mock_redis_client: aioredis.Redis,  # For completeness, though not strictly needed for close call
):
    """Test close_redis_client when _REDIS_CLIENT is already None."""
    # Ensure _REDIS_CLIENT is None
    with (
        patch("src.api.routes.jobs._REDIS_CLIENT", None),
        patch("src.api.routes.jobs.logger.info") as mock_logger_info,
    ):

        from src.api.routes import jobs as jobs_module

        assert jobs_module._REDIS_CLIENT is None  # Pre-condition

        await close_redis_client()

        # mock_redis_client.close should NOT have been called if the global was None
        # However, we are patching the global _REDIS_CLIENT, not what mock_redis_client is.
        # So we need to check that the *actual* client's close() wasn't called.
        # Since _REDIS_CLIENT was None, no actual close() on an object should occur.
        # The mock_redis_client fixture's close method is what we would check if it were _THE_ client.
        # Instead, we confirm no attempt to call .close() on a None object happened implicitly.

        # Verify logger was NOT called
        mock_logger_info.assert_not_called()

        # Verify _REDIS_CLIENT remains None
        assert jobs_module._REDIS_CLIENT is None
