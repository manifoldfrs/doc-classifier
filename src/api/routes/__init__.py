"""src/api/routes/__init__.py
###############################################################################
FastAPI **router registry**.
###############################################################################
This sub-package collates all APIRouter instances and exposes a convenience
``register_routes(app)`` helper so the central :pymod:`src.api.app` factory can
include them without hard-coding individual imports.  Each new route module
(e.g. *admin.py*, *jobs.py*) MUST:

1. Define a module-level variable named ``router`` of type ``fastapi.APIRouter``.
2. Add its router to :pydata:`ROUTERS` below by *import side-effect* (see
   ``from . import files as _files``).

This pattern keeps the *app* bootstrap logic declarative: future additions only
require editing **this** file, not the app factory.
"""

from __future__ import annotations

# third-party
from fastapi import APIRouter, FastAPI

from . import admin as _admin  # noqa: F401 – imported for side-effects only

# ---------------------------------------------------------------------------
# Import routers – **strict order** is not important for independent routes.
# ---------------------------------------------------------------------------
from . import files as _files  # noqa: F401  – imported for side-effects only

ROUTERS: list[APIRouter] = [
    _files.router,
    _admin.router,
]

__all__: list[str] = [
    "register_routes",
    "ROUTERS",
]


def register_routes(app: FastAPI) -> None:  # noqa: D401 – imperative helper
    """Attach every router in :pydata:`ROUTERS` onto **app**.

    The function is idempotent – attempting to include the *same* router twice
    is silently ignored by FastAPI.
    """

    for router in ROUTERS:
        app.include_router(router)
