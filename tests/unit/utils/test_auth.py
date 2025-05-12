from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException, Request

from src.utils.auth import verify_api_key
from tests.conftest import MockSettings


@pytest.fixture
def mock_request() -> MagicMock:
    """Provides a mock FastAPI Request object."""
    request = MagicMock(spec=Request)
    request.url = MagicMock()
    request.url.path = "/test/path"
    request.headers = {}  # Default to no headers
    request.state = MagicMock()  # Ensure request.state exists
    request.state.user = None  # Initialize user state
    return request


@pytest.mark.asyncio
async def test_verify_api_key_valid(mock_request: MagicMock) -> None:
    """Test `verify_api_key` with a valid API key."""
    settings = MockSettings(allowed_api_keys=["valid-key"])
    api_key_to_test = "valid-key"

    returned_key = await verify_api_key(api_key_to_test, mock_request, settings)

    assert returned_key == "valid-key"
    assert mock_request.state.user == "api_key_user"


@pytest.mark.asyncio
async def test_verify_api_key_invalid(mock_request: MagicMock) -> None:
    """Test `verify_api_key` with an invalid API key."""
    settings = MockSettings(allowed_api_keys=["valid-key"])
    api_key_to_test = "invalid-key"

    with pytest.raises(HTTPException) as exc_info:
        await verify_api_key(api_key_to_test, mock_request, settings)

    assert exc_info.value.status_code == 401
    assert "Invalid or missing x-api-key header" in exc_info.value.detail
    assert mock_request.state.user is None  # User should not be set on failure


@pytest.mark.asyncio
async def test_verify_api_key_missing(mock_request: MagicMock) -> None:
    """Test `verify_api_key` with a missing API key (header not provided)."""
    settings = MockSettings(allowed_api_keys=["valid-key"])
    api_key_to_test = None  # Simulates Header(None) when key is not sent

    with pytest.raises(HTTPException) as exc_info:
        await verify_api_key(api_key_to_test, mock_request, settings)

    assert exc_info.value.status_code == 401
    assert "Invalid or missing x-api-key header" in exc_info.value.detail
    assert mock_request.state.user is None


@pytest.mark.asyncio
async def test_verify_api_key_auth_disabled(mock_request: MagicMock) -> None:
    """Test `verify_api_key` when authentication is disabled (no keys in settings)."""
    settings = MockSettings(allowed_api_keys=[])  # Auth disabled
    api_key_to_test = "any-key-or-none"  # Value doesn't matter if auth is off

    # Test with a key provided
    returned_key_with_header = await verify_api_key(
        api_key_to_test, mock_request, settings
    )
    assert returned_key_with_header is None  # Returns None when auth disabled
    assert (
        mock_request.state.user is None
    )  # User should not be set if auth is truly disabled

    # Test with no key provided
    mock_request.state.user = None  # Reset for this sub-test
    returned_key_no_header = await verify_api_key(None, mock_request, settings)
    assert returned_key_no_header is None
    assert mock_request.state.user is None


@pytest.mark.asyncio
async def test_extract_api_key_dependency() -> None:
    """
    Tests the internal _extract_api_key dependency behavior.
    This is not directly testing verify_api_key but its helper.
    """
    from src.utils.auth import _extract_api_key

    # Simulate providing the header
    result_with_header = await _extract_api_key(x_api_key="test-header-key")
    assert result_with_header == "test-header-key"

    # Simulate header not provided
    result_no_header = await _extract_api_key(x_api_key=None)
    assert result_no_header is None
