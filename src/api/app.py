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
from fastapi import APIRouter, FastAPI
from fastapi.middleware.wsgi import WSGIMiddleware

# local imports
from src.api.errors import add_exception_handlers
from src.api.routes import admin as admin_router_module
from src.api.routes import files as files_router_module
from src.api.routes import jobs as jobs_router_module
from src.app import flask_app
from src.core.config import get_settings
from src.core.logging import RequestLoggingMiddleware, configure_logging

# The Prometheus instrumentation library is an optional dependency enabled via
# the PROMETHEUS_ENABLED environment variable (see src/core/config.py).  We
# wrap the import in a *try/except* so that the application still boots even
# when the package is not installed (e.g. stripped-down CI images).  The
# variable _PROM_AVAILABLE is used later to guard the actual instrumentation
# call so we avoid NameError in the except pathway.

try:
    from prometheus_fastapi_instrumentator import Instrumentator  # type: ignore

    _PROM_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover – optional dependency missing
    _PROM_AVAILABLE = False

__all__: list[str] = ["app"]

# ---------------------------------------------------------------------------
# Initialise *process-wide* logging before any logger instantiation.
# ---------------------------------------------------------------------------
settings = get_settings()
configure_logging(settings.debug)
logger = structlog.get_logger(__name__)


def _register_routes(app_instance: FastAPI) -> None:
    """Includes all defined API routers into the FastAPI application.

    This function centralizes the registration of routers, ensuring a clean
    setup process for the application. Each router is imported from its
    respective module within `src.api.routes`.
    """
    routers: list[APIRouter] = [
        files_router_module.router,
        admin_router_module.router,
        jobs_router_module.router,
    ]
    for router in routers:
        app_instance.include_router(router)


def _create_fastapi_app() -> FastAPI:  # noqa: D401 – factory
    """Build and configure the FastAPI application."""

    app_instance = FastAPI(
        title="HeronAI Document Classifier",
        version="0.1.0",
        docs_url="/docs",
        redoc_url=None,
        openapi_url="/openapi.json",
    )

    # ------------------------------------------------------------------
    # Middleware – logging comes first so later handlers inherit context vars.
    # ------------------------------------------------------------------
    app_instance.add_middleware(RequestLoggingMiddleware)

    # ------------------------------------------------------------------
    # Mount legacy Flask application under */legacy* for continuity.
    # ------------------------------------------------------------------
    app_instance.mount("/legacy", WSGIMiddleware(flask_app))

    # ------------------------------------------------------------------
    # Lifespan events
    # ------------------------------------------------------------------
    @app_instance.on_event("startup")
    async def _on_startup() -> None:  # pragma: no cover – trivial logging
        logger.info("fastapi_startup", commit_sha=settings.commit_sha)

    @app_instance.on_event("shutdown")
    async def _on_shutdown() -> None:  # pragma: no cover – trivial logging
        logger.info("fastapi_shutdown")

    # ------------------------------------------------------------------
    # Root endpoint – minimal landing page.  More detailed operational routes
    # (health, version) now live in ``src.api.routes.admin``.
    # ------------------------------------------------------------------
    @app_instance.get("/", include_in_schema=False)
    async def root() -> dict[str, str]:  # noqa: D401
        return {"message": "HeronAI Document Classifier – FastAPI layer"}

    # ------------------------------------------------------------------
    # Register all versioned API routers (files, admin, etc.)
    # ------------------------------------------------------------------
    _register_routes(app_instance)
    # ------------------------------------------------------------------
    # Register global exception handlers (HTTP 4xx/5xx → JSON envelope)
    # ------------------------------------------------------------------
    add_exception_handlers(app_instance)

    # ------------------------------------------------------------------
    # Prometheus metrics – optional and gated behind env var to keep the
    # footprint small in environments where metrics scraping is not desired
    # (e.g. unit-test runners).  When enabled the Instrumentator adds rich
    # default metrics (latency histograms, request counts, in-progress gauge)
    # and exposes them under **/metrics** which is *excluded* from the OpenAPI
    # schema as per best practices.
    # ------------------------------------------------------------------
    if settings.prometheus_enabled and _PROM_AVAILABLE:  # pragma: no cover –
        Instrumentator().instrument(app_instance).expose(  # noqa: WPS437 fluent chain
            app_instance,
            endpoint="/metrics",
            include_in_schema=False,
        )
        logger.info("prometheus_instrumentation_enabled")
    elif settings.prometheus_enabled and not _PROM_AVAILABLE:
        logger.warning(
            "prometheus_instrumentation_requested_but_package_missing",
            advice="Add 'prometheus-fastapi-instrumentator' to requirements.txt",
        )

    return app_instance


# Instantiate once at import time.
app: FastAPI = _create_fastapi_app()
