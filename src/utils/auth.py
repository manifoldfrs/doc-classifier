from __future__ import annotations

from typing import Annotated, Optional

import structlog
from fastapi import Depends, Header, HTTPException, Request, status

from src.core.config import Settings, get_settings

__all__: list[str] = [
    "verify_api_key",
]

logger = structlog.get_logger(__name__)


async def _extract_api_key(
    x_api_key: Optional[str] = Header(None),
) -> Optional[str]:
    """Return the raw ``x-api-key`` header if present (case-insensitive)."""

    return x_api_key  # FastAPI handles header name case-insensitively


async def verify_api_key(
    api_key: Annotated[Optional[str], Depends(_extract_api_key)],
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
) -> str | None:
    """Verifies the provided API key against configured allowed keys.

    Args:
        api_key: The API key extracted from the 'x-api-key' header.
        request: The FastAPI request object.
        settings: Application settings, including allowed API keys.

    Returns:
        The validated API key if successful.

    Raises:
        HTTPException (401): If auth is enabled and the key is missing or invalid.
    """

    allowed = settings.allowed_api_keys
    if not allowed:  # Auth disabled – log once at DEBUG level and continue
        logger.debug("auth_skipped", reason="no_keys_configured")
        return None

    if api_key is None or api_key not in allowed:
        logger.warning(
            "auth_failed",
            path=request.url.path,
            has_header=api_key is not None,
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing x-api-key header.",
        )

    request.state.user = "api_key_user"  # TODO: placeholder until RBAC extension
    logger.debug("auth_success", path=request.url.path)
    return api_key
