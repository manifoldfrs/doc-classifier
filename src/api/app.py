"""HeronAI ─ FastAPI application
================================

This module hosts the **production ASGI application**.  It was extracted from
``src/api/__init__.py`` to comply with the project rule that *__init__.py*
files remain minimal (see `.cursorrules` §3 → 9).

Usage
-----
Run locally with::

    uvicorn src.api.app:app --reload

The FastAPI instance is exposed as ``app``.  A thin re-export remains in
``src/api/__init__.py`` so that existing references to ``src.api:app`` do not
break, but all implementation now lives here.
"""

from __future__ import annotations

# third-party
import structlog  # type: ignore
from fastapi import FastAPI
from fastapi.middleware.wsgi import WSGIMiddleware
from fastapi.responses import JSONResponse

# local
from src.app import flask_app  # Legacy WSGI application
from src.core.config import get_settings

__all__: list[str] = ["app"]

logger = structlog.get_logger(__name__)
settings = get_settings()


def _create_fastapi_app() -> FastAPI:  # noqa: D401 – factory
    """Build and configure the FastAPI application."""

    app = FastAPI(
        title="HeronAI Document Classifier",
        version="0.1.0",
        docs_url="/docs",
        redoc_url=None,
        openapi_url="/openapi.json",
    )

    # ------------------------------------------------------------------
    # Mount legacy Flask application
    # ------------------------------------------------------------------
    app.mount("/legacy", WSGIMiddleware(flask_app))

    # ------------------------------------------------------------------
    # Lifespan events
    # ------------------------------------------------------------------
    @app.on_event("startup")
    async def _on_startup() -> None:  # pragma: no cover – trivial logging
        logger.info("fastapi_startup", commit_sha=settings.commit_sha)

    @app.on_event("shutdown")
    async def _on_shutdown() -> None:  # pragma: no cover – trivial logging
        logger.info("fastapi_shutdown")

    # ------------------------------------------------------------------
    # Temporary health endpoint (to be superseded by admin router)
    # ------------------------------------------------------------------
    @app.get("/v1/health", tags=["admin"], summary="Lightweight health probe")
    async def health() -> JSONResponse:  # noqa: D401
        """Return **200 OK** if the service is alive."""

        return JSONResponse({"status": "ok", "commit_sha": settings.commit_sha})

    @app.get("/", include_in_schema=False)
    async def root() -> dict[str, str]:  # noqa: D401
        return {"message": "HeronAI Document Classifier – FastAPI layer"}

    return app


# Instantiate once at import time.
app: FastAPI = _create_fastapi_app()
