"""HeronAI API package root.

The actual FastAPI application lives in :pymod:`src.api.app`.  We re-export the
``app`` variable here solely for backwards-compatibility with entry-points that
expect ``src.api:app``.
"""

from __future__ import annotations

from importlib import import_module as _import_module

# Perform *lazy* import so that loading the package itself does not incur the
# cost of building the FastAPI application – it will only be constructed when
# something accesses the attribute.  This keeps module import times low for
# tooling such as ``pytest --collect-only`` while still offering ergonomic
# access via ``src.api:app``.
_app_module = _import_module(".app", package=__name__)
app = _app_module.app  # type: ignore[attr-defined]
