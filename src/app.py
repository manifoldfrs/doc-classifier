"""Legacy Flask application
==========================

This module contains the *original* Flask implementation that powered the
prototype endpoint.  While the project transitions to a FastAPI architecture
(`src/api`), we keep the Flask routes available under the **/legacy** prefix to
avoid breaking consumers that may still depend on them.

Key points:
• The Flask instance is now exposed as **flask_app** – the canonical symbol
  referenced by the FastAPI layer via Starlette's ``WSGIMiddleware``.
• A secondary alias **app** is retained *temporarily* to avoid breaking the
  existing unit tests in ``tests/test_app.py``.  This alias will be removed
  once the test suite is migrated in *Implementation Plan » Step 9*.
• All environment-driven configuration logic has been left intact so that the
  behaviour of the legacy endpoint does not change.
"""

from __future__ import annotations

# stdlib
import os
from typing import Final

# third-party
from dotenv import load_dotenv  # type: ignore[import-not-found]
from flask import Flask, Response, jsonify, request  # type: ignore[import-not-found]

# local
from src.classifier import classify_file

# ---------------------------------------------------------------------------
# Environment / configuration
# ---------------------------------------------------------------------------
load_dotenv()

ALLOWED_EXTENSIONS: Final[set[str]] = {
    ext.strip().lower()
    for ext in (
        os.getenv("ALLOWED_EXTENSIONS")
        or "pdf,docx,doc,xlsx,xlsb,xls,csv,jpg,jpeg,png,txt,md,xml,json,html,eml"
    ).split(",")
    if ext.strip()
}


# ---------------------------------------------------------------------------
# Flask application factory (simple singleton suffices for demo)
# ---------------------------------------------------------------------------
flask_app = Flask(__name__)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def allowed_file(filename: str) -> bool:  # noqa: D401 – simple helper
    """Return *True* if **filename** has an allowed extension.

    The check is **case-insensitive** and expects a period separating the base
    name and extension (e.g. ``"document.pdf"``).
    """

    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ---------------------------------------------------------------------------
# Routes – kept minimal as new functionality moves to FastAPI
# ---------------------------------------------------------------------------
@flask_app.route("/classify_file", methods=["POST"])
def classify_file_route() -> Response:  # noqa: D401 – Flask view function
    """Classify a single file based on *heuristics* in ``src.classifier``.

    The endpoint mirrors the original behaviour for continuity.  Validation is
    intentionally basic – richer checks will be introduced in the FastAPI
    rewrite.
    """

    if "file" not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": f"File type not allowed: {file.filename}"}), 400

    file_class = classify_file(file)
    return jsonify({"file_class": file_class}), 200


# ---------------------------------------------------------------------------
# Temporary backwards-compat alias – will be removed in Step 9
# ---------------------------------------------------------------------------
app = flask_app  # noqa: N816 – alias kept for test compatibility


# ---------------------------------------------------------------------------
# Local development entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover – manual dev run only
    flask_app.run(debug=True)
