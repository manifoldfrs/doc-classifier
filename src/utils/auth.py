###############################################################################
# src/utils/auth.py
# -----------------------------------------------------------------------------
# API-key authentication helpers (Implementation Plan – Step 6.1)
#
# This module centralises **header-based** authentication for every FastAPI
# route in the HeronAI demo.  The specification (§6 *Authentication &
# Authorization*) mandates that clients supply an ``x-api-key`` request header
# which MUST match one of the comma-separated secrets configured in the
# ``ALLOWED_API_KEYS`` environment variable.
#
# Design considerations
# =====================
# 1. **Zero globals** – we rely solely on dependency-injection via FastAPI's
#    ``Depends``.  No state is stored at module scope.
# 2. **Early-exit** – when the **allowed_api_keys** list is *empty* the service
#    operates in *open* mode, equivalent to authentication being disabled.
#    This is convenient in local development environments where setting up
#    secrets is cumbersome.
# 3. **Structured logging** – failures are logged with *structlog* for
#    observability, but sensitive key material is *never* included to avoid
#    leaking secrets.
# 4. **Typed return value** – the dependency returns the validated API key so
#    downstream path-operations can access ``request.state.user`` or similar in
#    future extensions (multi-tenancy, RBAC).
###############################################################################

from __future__ import annotations

# stdlib
from typing import Annotated, Optional

# third-party
import structlog
from fastapi import Depends, Header, HTTPException, Request, status

# local
from src.core.config import Settings, get_settings

__all__: list[str] = [
    "verify_api_key",
]

logger = structlog.get_logger(__name__)


async def _extract_api_key(
    x_api_key: Optional[str] = Header(None),
) -> Optional[str]:  # noqa: D401 – internal helper
    """Return the raw ``x-api-key`` header if present (case-insensitive)."""

    return x_api_key  # FastAPI handles header name case-insensitively


async def verify_api_key(
    api_key: Annotated[Optional[str], Depends(_extract_api_key)],
    request: Request,
    settings: Settings = Depends(get_settings),  # noqa: B008
) -> str | None:  # noqa: D401 – dependency signature
    """Validate *api_key* against the allowed list declared in **settings**.

    Behaviour
    ---------
    • When **settings.allowed_api_keys** is **empty** the dependency *always*
      succeeds, effectively disabling authentication (useful for local dev).
    • When the header is **missing** or **invalid** → raises **401** with a
      stable error envelope handled by the global exception middleware.

    The function returns the *validated* API key so that downstream handlers
    can bind it to *structlog* or attach it to ``request.state.user`` once the
    multi-tenancy extension (spec §14) is implemented.
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

    # Successful validation – bind user context for structured logs
    request.state.user = "api_key_user"  # placeholder until RBAC extension
    logger.debug("auth_success", path=request.url.path)
    return api_key
