from __future__ import annotations

from importlib import import_module as _import_module

_app_module = _import_module(".app", package=__name__)
app = _app_module.app
