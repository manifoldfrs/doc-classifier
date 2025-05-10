from __future__ import annotations

# Re-export *Settings* model and lazy singleton accessor for ergonomic imports.
from .config import Settings, get_settings  # noqa: F401

__all__: list[str] = [
    "Settings",
    "get_settings",
]
