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
import structlog
from fastapi import FastAPI
from fastapi.middleware.wsgi import WSGIMiddleware

# local imports
from src.api.errors import add_exception_handlers
from src.api.routes import register_routes  # local import to avoid cycles
from src.app import flask_app
from src.core.config import get_settings
from src.core.logging import RequestLoggingMiddleware, configure_logging

__all__: list[str] = ["app"]

# ---------------------------------------------------------------------------
# Initialise *process-wide* logging before any logger instantiation.
# ---------------------------------------------------------------------------
settings = get_settings()
configure_logging(settings.debug)
logger = structlog.get_logger(__name__)


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
    # Middleware – logging comes first so later handlers inherit context vars.
    # ------------------------------------------------------------------
    app.add_middleware(RequestLoggingMiddleware)

    # ------------------------------------------------------------------
    # Mount legacy Flask application under */legacy* for continuity.
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
    # Root endpoint – minimal landing page.  More detailed operational routes
    # (health, version) now live in ``src.api.routes.admin``.
    # ------------------------------------------------------------------
    @app.get("/", include_in_schema=False)
    async def root() -> dict[str, str]:  # noqa: D401
        return {"message": "HeronAI Document Classifier – FastAPI layer"}

    # ------------------------------------------------------------------
    # Register all versioned API routers (files, admin, etc.)
    # ------------------------------------------------------------------
    register_routes(app)
    # ------------------------------------------------------------------
    # Register global exception handlers (HTTP 4xx/5xx → JSON envelope)
    # ------------------------------------------------------------------
    add_exception_handlers(app)

    return app


# Instantiate once at import time.
app: FastAPI = _create_fastapi_app()
