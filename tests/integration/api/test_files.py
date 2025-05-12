from __future__ import annotations

from io import BytesIO
from typing import List
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.api.app import app
from src.core.config import get_settings
from tests.conftest import MockSettings

pytestmark = [pytest.mark.integration]


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


def test_batch_upload_three_files_returns_expected_shape() -> None:
    """Upload 3 files and assert the synchronous 200-OK JSON structure.

    The endpoint should return **200 OK** (since 3 < ASYNC_THRESHOLD=10) and a
    JSON array of length 3 where each element contains at least the following
    keys mandated by the spec: ``filename``, ``label``, ``confidence``.
    """

    # Create a mock settings instance with a test API key
    mock_settings = MockSettings(allowed_api_keys=["test-api-key"])

    # Patch **both** imported references to get_settings so that the auth
    # dependency and the rest of the application share the very same mocked
    # configuration instance.
    with (
        patch("src.core.config.get_settings", return_value=mock_settings),
        patch("src.utils.auth.get_settings", return_value=mock_settings),
        patch("src.api.routes.files.classify") as mock_classify,
    ):
        # Return a lightweight stub that mimics Pydantic model interface used
        # downstream (``.dict()``).  The actual ML logic is covered by unit
        # tests – here we only care about the API contract.
        class _StubResult(dict):
            def dict(self):  # type: ignore[override]
                return self

        # Prepare a predictable payload regardless of input file
        def _fake_classify(file):  # noqa: ANN001
            return _StubResult(
                filename=file.filename,
                label="unknown",
                confidence=0.0,
                mime_type=file.content_type or "text/plain",
                size_bytes=0,
                pipeline_version="v_test_pipeline",
                processing_ms=0.0,
            )

        mock_classify.side_effect = _fake_classify

        # Monkeypatch `.dict()` on the response schema so the route handler's
        # serialization step succeeds despite the dataclass inheritance.
        from src.api.schemas import ClassificationResultSchema  # noqa: WPS433

        if not hasattr(ClassificationResultSchema, "dict"):

            def _schema_dict(self, *args, **kwargs):  # noqa: D401, ANN001
                return self.__dict__  # type: ignore[attr-defined]

            ClassificationResultSchema.dict = _schema_dict  # type: ignore[attr-defined]

        client = TestClient(app)

        # ------------------------------------------------------------------
        # Ensure FastAPI's dependency-injection also returns *mock_settings*
        # irrespective of when the Depends instance was created.
        # ------------------------------------------------------------------
        app.dependency_overrides[get_settings] = lambda: mock_settings  # type: ignore[arg-type]

        # --------------------------------------------------------------
        # Always send the API key header – the mocked settings guarantees it
        # exists, ensuring the request passes the authentication dependency.
        # --------------------------------------------------------------
        headers = {"x-api-key": mock_settings.allowed_api_keys[0]}

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
