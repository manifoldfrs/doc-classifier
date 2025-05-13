import pytest
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
from pandas.errors import EmptyDataError

# ---------------------------------------------------------------------------
# metadata stage tests (classification.stages.metadata)
# ---------------------------------------------------------------------------
from src.classification.stages import metadata as _metadata_mod
from src.classification.stages.metadata import stage_metadata
from src.classification.types import StageOutcome


@pytest.mark.asyncio
async def test_stage_metadata_pdf_match(monkeypatch: pytest.MonkeyPatch) -> None:
    """PDF files should yield a label when metadata patterns match."""

    # Short-circuit the heavyweight PDF text extraction with a stub
    monkeypatch.setattr(
        _metadata_mod,
        "_extract_pdf_metadata",
        AsyncMock(return_value="This contract agreement is legally binding."),
    )

    mock_file = MagicMock()
    mock_file.filename = "contract.pdf"
    mock_file.content_type = "application/pdf"
    mock_file.seek = AsyncMock()
    mock_file.read = AsyncMock(return_value=b"%PDF-1.4 dummy")

    outcome: StageOutcome = await stage_metadata(mock_file)
    assert outcome.label == "contract"
    # Confidence comes from METADATA_PATTERNS mapping (0.85)
    assert pytest.approx(outcome.confidence) == 0.85


@pytest.mark.asyncio
async def test_stage_metadata_skip_non_pdf() -> None:
    """Non-PDF files must be skipped with a null outcome."""
    mock_file = MagicMock()
    mock_file.filename = "image.png"
    mock_file.content_type = "image/png"
    # .seek/read should never be awaited for skipped files, but add dummies anyway
    mock_file.seek = AsyncMock()
    mock_file.read = AsyncMock(return_value=b"")

    outcome = await stage_metadata(mock_file)
    assert outcome.label is None and outcome.confidence is None


@pytest.mark.asyncio
async def test_stage_metadata_handles_exceptions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Any unexpected error inside the stage should be swallowed and converted to a null outcome."""

    # Force the helper to raise to exercise the exception branch
    async def _boom(_: bytes) -> str:  # pragma: no cover – body executed in test
        raise RuntimeError("explode")

    monkeypatch.setattr(_metadata_mod, "_extract_pdf_metadata", _boom)

    mock_file = MagicMock()
    mock_file.filename = "bad.pdf"
    mock_file.content_type = "application/pdf"
    mock_file.seek = AsyncMock()
    mock_file.read = AsyncMock(return_value=b"%PDF bad")

    outcome = await stage_metadata(mock_file)
    assert outcome == StageOutcome(label=None, confidence=None)


# ---------------------------------------------------------------------------
# CSV parsing tests (parsing.csv)
# ---------------------------------------------------------------------------
from src.parsing.csv import extract_text_from_csv as _extract_csv_text


@pytest.mark.asyncio
async def test_extract_text_from_csv_happy() -> None:
    """Well-formed CSV should be converted to space-separated text."""
    csv_bytes = b"a,b\n1,2\n3,4\n"

    mock_file = MagicMock()
    mock_file.seek = AsyncMock()
    mock_file.read = AsyncMock(return_value=csv_bytes)

    text = await _extract_csv_text(mock_file)
    assert text.strip() == "a b\n1 2\n3 4"


@pytest.mark.asyncio
async def test_extract_text_from_csv_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Malformed CSV must fall back to raw UTF-8 decoding (with replacement)."""

    csv_bytes = b"bad,\xff,data"  # invalid UTF-8 sequence to trigger replacement char

    # Patch pandas.read_csv to raise EmptyDataError so the fallback path is taken
    def _raise(*_: bytes, **__: bytes):  # noqa: D401 – simple stub signature
        raise EmptyDataError("no data")

    monkeypatch.setattr(pd, "read_csv", _raise)

    mock_file = MagicMock()
    mock_file.seek = AsyncMock()
    mock_file.read = AsyncMock(return_value=csv_bytes)

    text = await _extract_csv_text(mock_file)
    # Replacement character � (0xFFFD) will appear where invalid byte was located
    assert "bad" in text and "data" in text
