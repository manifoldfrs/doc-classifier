"""src/api/routes/__init__.py
###############################################################################
FastAPI **router package marker**.
###############################################################################
This sub-package collates all APIRouter instances. The actual registration of
these routers into the FastAPI application now happens directly in
`src.api.app.py`.

Each route module (e.g. *admin.py*, *jobs.py*) MUST:
1. Define a module-level variable named ``router`` of type ``fastapi.APIRouter``.

This __init__.py file is kept minimal, primarily acting as a Python package
identifier and a point for potential future shared route utilities if needed,
though direct imports into `src.api.app.py` are now preferred for routers.
"""

from __future__ import annotations

# This file is intentionally kept minimal.
# Routers are imported and registered directly in src/api/app.py.

__all__: list[str] = []
