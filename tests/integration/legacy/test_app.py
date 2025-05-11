"""tests/integration/legacy/test_app.py
###############################################################################
Legacy Flask route tests – relocated from *tests/test_app.py* (Plan Step 9.1)
###############################################################################
These tests validate the original Flask implementation that is now exposed
under the **/legacy** mount path of the FastAPI application.  They remain in
place to guarantee backward-compatibility while new clients migrate to the
FastAPI endpoints.

Markers
=======
* ``integration`` – full-stack tests that spin up the WSGI app.
* ``legacy`` – subset tag specific to deprecated Flask routes.

Run **only** these tests via::

    pytest -m "integration and legacy"
"""

from io import BytesIO
from typing import Set

import pytest

# ---------------------------------------------------------------------------
# Pytest markers applied module-wide so individual tests do not need to repeat
# them.  Developers can still override at the function level if required.
# ---------------------------------------------------------------------------
pytestmark = [
    pytest.mark.integration,
    pytest.mark.legacy,
]

# ---------------------------------------------------------------------------
# Ensure **ALLOWED_EXTENSIONS** constant in *src.app* includes the extensions
# required by these tests.  Some unit-tests manipulate the ALLOWED_EXTENSIONS
# environment variable, and because *src.app* evaluates this at import-time we
# patch the module attribute directly.
# ---------------------------------------------------------------------------
from src import app as app_module  # noqa: E402 – imported after env safeguards
from src.app import allowed_file, app  # noqa: E402 – import after pytestmark


@pytest.fixture(autouse=True)
def _patch_allowed_extensions(monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: D401
    """Ensure legacy `allowed_file()` uses a permissive extension set."""

    default_exts: Set[str] = {
        "pdf",
        "png",
        "jpg",
        "jpeg",
        "txt",
        "doc",
        "docx",
        "xls",
        "xlsx",
        "csv",
    }

    monkeypatch.setattr(app_module, "ALLOWED_EXTENSIONS", default_exts, raising=False)

    # Also align environment variable so any *re-import* of src.app gets the
    # same permissive set.
    monkeypatch.setenv("ALLOWED_EXTENSIONS", ",".join(sorted(default_exts)))

    yield


@pytest.fixture
def client():
    """Return a Flask *test_client* for the legacy application."""

    app.config["TESTING"] = True
    with app.test_client() as test_client:
        yield test_client


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("file.pdf", True),
        ("file.png", True),
        ("file.jpg", True),
        ("file.txt", True),
        ("file", False),
    ],
)
def test_allowed_file(filename: str, expected: bool) -> None:
    """`allowed_file()` should accept/reject extensions correctly."""

    assert allowed_file(filename) is expected


def test_no_file_in_request(client):
    """POST without multipart **file** field should yield 400."""

    response = client.post("/classify_file")
    assert response.status_code == 400


def test_no_selected_file(client):
    """An empty filename should trigger a 400 error."""

    data = {"file": (BytesIO(b""), "")}  # Empty filename
    response = client.post(
        "/classify_file", data=data, content_type="multipart/form-data"
    )
    assert response.status_code == 400


def test_success(client, mocker):
    """Happy-path – classifier returns JSON payload with *file_class*."""

    mocker.patch("src.app.classify_file", return_value="test_class")

    data = {"file": (BytesIO(b"dummy content"), "file.pdf")}
    response = client.post(
        "/classify_file", data=data, content_type="multipart/form-data"
    )
    assert response.status_code == 200
    assert response.get_json() == {"file_class": "test_class"}
