"""tests/integration/api/test_files.py
###############################################################################
Integration tests for the **/v1/files** FastAPI endpoint (Implementation Plan –
Step 9.3).
###############################################################################
The goal is **not** to exhaustively test the classification pipeline – that is
covered by unit-tests in *tests/unit/*.  Here we simply validate the public
contract exposed by the upload endpoint:

1. A batch upload of *≤ ASYNC_THRESHOLD* files returns synchronous **200 OK**.
2. The JSON payload is an **array** whose length matches the number of files
   supplied.
3. Every element contains the *core* keys required by the technical
   specification so front-ends can rely on a stable schema.

The test spins up an in-process TestClient so **no network** or separate server
process is required.  This keeps the suite deterministic and avoids race
conditions in CI.  The files are generated **in-memory** (via ``io.BytesIO``),
thus no fixture data is read from disk which speeds up execution.

Pytest Markers
==============
We apply the *integration* marker at module scope so developers can include or
exclude these tests via ``-m integration``.
"""

from __future__ import annotations

# stdlib
from io import BytesIO
from typing import List

# third-party
import pytest
from fastapi.testclient import TestClient

# local – import AFTER pytestmark so marker registration happens first
from src.api.app import app  # noqa: E402 – needed for TestClient
from src.core.config import get_settings  # NEW – to obtain allowed_api_keys

###############################################################################
# Pytest markers – module-level so all tests inherit them automatically.
###############################################################################
pytestmark = [pytest.mark.integration]

###############################################################################
# Helper utilities
###############################################################################


def _build_multipart_payload() -> List[tuple[str, tuple[str, BytesIO, str]]]:
    """Return a **files** payload suitable for ``client.post(..., files=...)``.

    The helper constructs *three* small text files with extensions that pass
    the validators:

    • ``invoice_123.txt`` – contains the keyword *invoice* so the filename stage
      should classify it as *invoice*.
    • ``bank_statement.txt`` – should map to *bank_statement*.
    • ``random_notes.txt`` – intentionally ambiguous, expected label *unknown*.

    Using plain text keeps the payload trivial and avoids dependencies on
    external parsers (pdfminer, Pillow, etc.) for this integration test.
    """

    samples = [
        ("invoice_123.txt", b"Invoice #123 amount due", "text/plain"),
        ("bank_statement.txt", b"Bank statement content", "text/plain"),
        ("random_notes.txt", b"Some random content", "text/plain"),
    ]
    return [
        ("files", (name, BytesIO(content), mime)) for name, content, mime in samples
    ]


###############################################################################
# Tests
###############################################################################


def test_batch_upload_three_files_returns_expected_shape() -> None:
    """Upload 3 files and assert the synchronous 200-OK JSON structure.

    The endpoint should return **200 OK** (since 3 < ASYNC_THRESHOLD=10) and a
    JSON array of length 3 where each element contains at least the following
    keys mandated by the spec: ``filename``, ``label``, ``confidence``.
    """

    client = TestClient(app)

    # --------------------------------------------------------------
    # Some deployments configure static API keys.  When the list is
    # non-empty we must supply a valid header to avoid 401 responses.
    # --------------------------------------------------------------
    settings = get_settings()
    headers = {}
    if settings.allowed_api_keys:
        headers = {"x-api-key": settings.allowed_api_keys[0]}

    response = client.post(
        "/v1/files", files=_build_multipart_payload(), headers=headers
    )

    # 1. HTTP-layer assertions -------------------------------------------------
    assert response.status_code == 200, response.text
    assert response.headers.get("content-type", "").startswith("application/json")

    # 2. Payload shape ---------------------------------------------------------
    payload = response.json()
    assert isinstance(payload, list), "Response root should be a JSON array"
    assert len(payload) == 3, "Array length must match number of uploaded files"

    required_keys = {"filename", "label", "confidence"}
    for item in payload:
        # Each item should be a dict with at least the required keys
        assert isinstance(item, dict), "Each element must be a JSON object"
        assert required_keys.issubset(
            item.keys()
        ), f"Missing keys {required_keys - item.keys()} in {item}"
        # Basic type checks – confidence is a float in [0, 1]
        assert isinstance(item["label"], str)
        assert 0.0 <= float(item["confidence"]) <= 1.0

    # 3. Correlation header ----------------------------------------------------
    # The service includes `X-Request-ID` on both request logging and response.
    assert "X-Request-ID" in response.headers
