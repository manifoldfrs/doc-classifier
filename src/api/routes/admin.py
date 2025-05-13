from __future__ import annotations

from typing import Dict

import structlog
from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from src.core.config import Settings, get_settings
from src.utils.auth import verify_api_key

__all__: list[str] = [
    "router",
]

logger = structlog.get_logger(__name__)

router = APIRouter(
    prefix="/v1",
    tags=["Admin"],
    dependencies=[Depends(verify_api_key)],
)

SETTINGS_DEP: Settings = Depends(get_settings)


@router.get("/health", response_model=Dict[str, str])
async def health(
    settings: Settings = SETTINGS_DEP,
) -> Dict[str, str]:
    """Return service health status."""
    return {"status": "ok", "commit_sha": settings.commit_sha or "unknown"}


@router.get("/version", summary="Application version information")
async def version(
    request: Request,
    settings: Settings = SETTINGS_DEP,
) -> JSONResponse:
    """Return the semantic *application* version plus the git commit SHA.

    The FastAPI *application* instance exposes its declared version via
    ``request.app.version`` which we echo back in a stable JSON schema so that
    CI systems (or manual cURL) can easily parse it.
    """

    return JSONResponse(
        {
            "version": request.app.version,
            "commit_sha": settings.commit_sha,
        }
    )
