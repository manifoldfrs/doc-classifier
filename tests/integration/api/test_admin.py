from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.app import app
from src.core.config import get_settings
from tests.conftest import MockSettings

pytestmark = [pytest.mark.integration]


@pytest.fixture
def client(mock_settings: MockSettings) -> TestClient:
    """Provides a TestClient instance with overridden settings."""
    # Ensure the app uses the mocked settings
    app.dependency_overrides[get_settings] = lambda: mock_settings
    return TestClient(app)


def test_health_endpoint(client: TestClient, mock_settings: MockSettings) -> None:
    """
    Tests the /v1/health endpoint.
    It should return 200 OK with status 'ok' and commit_sha.
    """
    mock_settings.commit_sha = "test-commit-sha"
    mock_settings.allowed_api_keys = ["test-key"]

    response = client.get("/v1/health", headers={"x-api-key": "test-key"})

    assert response.status_code == 200
    payload = response.json()
    assert payload == {
        "status": "ok",
        "commit_sha": "test-commit-sha",
    }


def test_health_endpoint_no_auth(
    client: TestClient, mock_settings: MockSettings
) -> None:
    """
    Tests the /v1/health endpoint when no API key is required.
    """
    mock_settings.commit_sha = "test-commit-sha-no-auth"
    mock_settings.allowed_api_keys = []  # Auth disabled

    response = client.get("/v1/health")  # No API key header

    assert response.status_code == 200
    payload = response.json()
    assert payload == {
        "status": "ok",
        "commit_sha": "test-commit-sha-no-auth",
    }


def test_version_endpoint(client: TestClient, mock_settings: MockSettings) -> None:
    """
    Tests the /v1/version endpoint.
    It should return 200 OK with the application version and commit_sha.
    """
    mock_settings.commit_sha = "another-commit-sha"
    mock_settings.allowed_api_keys = ["test-key"]
    app.version = (
        "1.2.3-test"  # Set a specific version for the app instance for this test
    )

    response = client.get("/v1/version", headers={"x-api-key": "test-key"})

    assert response.status_code == 200
    payload = response.json()
    assert payload == {
        "version": "1.2.3-test",
        "commit_sha": "another-commit-sha",
    }
    # Clean up app.version if necessary or ensure it's reset if it's dynamic
    app.version = get_settings().pipeline_version  # Reset to original or a default


def test_admin_endpoints_unauthorized(
    client: TestClient, mock_settings: MockSettings
) -> None:
    """
    Tests admin endpoints return 401 if API key is required and not provided or invalid.
    """
    mock_settings.allowed_api_keys = ["secret-key"]

    response_health_no_key = client.get("/v1/health")
    assert response_health_no_key.status_code == 401

    response_version_wrong_key = client.get(
        "/v1/version", headers={"x-api-key": "wrong-key"}
    )
    assert response_version_wrong_key.status_code == 401
