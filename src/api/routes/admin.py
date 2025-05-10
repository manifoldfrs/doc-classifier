"""src/api/routes/admin.py
###############################################################################
Administrative and health-check routes (Implementation Plan – Step 5.2)
###############################################################################
This module provides a dedicated **APIRouter** exposing operational endpoints
that are required for production readiness and monitoring:

1. ``GET /v1/health`` – Liveness probe consumed by load-balancers and uptime
   checks.  Returns HTTP **200** together with a payload indicating the current
   git commit SHA so deployments can be correlated with revision history.
2. ``GET /v1/version`` – Human-friendly endpoint that exposes the semantic
   application version *and* the commit SHA.  This is useful for CI/CD
   pipelines and troubleshooting as it avoids parsing the OpenAPI schema.

Design notes
============
• The router lives in its own module to keep concerns *separated* from
  business-logic routes (file uploads, job polling, etc.).  This aligns with
  the repository's **Single Responsibility** rule.
• Endpoints are intentionally lightweight and avoid any blocking I/O; the
  service currently has no dependencies (DB, Redis) so expensive readiness
  checks are unnecessary.
• Both endpoints bind to the ``admin`` tag so that the generated OpenAPI
  documentation groups them together.
"""

from __future__ import annotations

# third-party
from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

# local
from src.core.config import get_settings
from src.utils.auth import verify_api_key

__all__: list[str] = [
    "router",
]

# ---------------------------------------------------------------------------
# Router instance – mounted with prefix "/v1" by the registration helper.
# ---------------------------------------------------------------------------

router: APIRouter = APIRouter(
    prefix="/v1",
    tags=["admin"],
    include_in_schema=True,
    dependencies=[Depends(verify_api_key)],
)

settings = get_settings()


@router.get("/health", summary="Liveness probe")
async def health() -> JSONResponse:  # noqa: D401 – FastAPI path operation
    """Return **200 OK** if the service process is responsive.

    The response payload purposefully remains minimal to keep the endpoint fast
    and to avoid leaking sensitive information.  Clients interested in richer
    metadata should call :http:get:`/v1/version` instead.
    """

    return JSONResponse(
        {
            "status": "ok",
            "commit_sha": settings.commit_sha,
        }
    )


@router.get("/version", summary="Application version information")
async def version(
    request: Request,
) -> JSONResponse:  # noqa: D401 – FastAPI path operation
    """Return the semantic *application* version plus the git commit SHA.

    The FastAPI *application* instance exposes its declared version via
    ``request.app.version`` which we echo back in a stable JSON schema so that
    CI systems (or manual cURL) can easily parse it.
    """

    return JSONResponse(
        {
            "version": request.app.version,  # set in ``src.api.app._create_fastapi_app``
            "commit_sha": settings.commit_sha,
        }
    )
