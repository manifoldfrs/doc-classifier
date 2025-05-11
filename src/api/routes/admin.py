from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from src.core.config import Settings, get_settings
from src.utils.auth import verify_api_key

__all__: list[str] = [
    "router",
]

router: APIRouter = APIRouter(
    prefix="/v1",
    tags=["admin"],
    include_in_schema=True,
    dependencies=[Depends(verify_api_key)],
)


@router.get("/health", summary="Liveness probe")
async def health(
    settings: Settings = Depends(get_settings),  # noqa: B008
) -> JSONResponse:  # noqa: D401 – FastAPI path operation
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
    settings: Settings = Depends(get_settings),  # noqa: B008
) -> JSONResponse:  # noqa: D401 – FastAPI path operation
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
