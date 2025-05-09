"""
Security utilities for the API.

This module provides security-related functionality such as API key authentication
that can be used as dependencies in FastAPI route handlers.
"""

from dotenv import load_dotenv
from fastapi import Security, HTTPException, status
from fastapi.security.api_key import APIKeyHeader

from src.core.config import settings
from src.core.logging_config import get_logger

# Initialize logger
logger = get_logger(__name__)

load_dotenv()


# Create API key header security scheme
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)


async def get_api_key(api_key: str = Security(api_key_header)) -> str:
    """
    Validate the API key provided in the request header.

    This dependency can be used to protect API endpoints that require authentication.
    It checks if the API key provided in the X-API-Key header is valid.

    Args:
        api_key: The API key extracted from the request header by the APIKeyHeader dependency.

    Returns:
        str: The validated API key if authentication is successful.

    Raises:
        HTTPException: If the API key is invalid or missing.
    """
    if api_key in settings.ALLOWED_API_KEYS:
        logger.debug("Valid API key provided")
        return api_key
    else:
        logger.warning(f"Invalid API key attempt: {api_key[:3]}...")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key",
        )
