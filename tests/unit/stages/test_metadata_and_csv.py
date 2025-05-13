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
from src.core.exceptions import MetadataProcessingError


@pytest.mark.asyncio
async def test_stage_metadata_pdf_match(monkeypatch: pytest.MonkeyPatch) -> None:
    """PDF files should yield a label when metadata patterns match."""

    # Mock _extract_pdf_metadata to return a specific string and check it's called correctly
    mocked_extraction_result = "This contract agreement is legally binding."
    mock_extract_metadata = AsyncMock(return_value=mocked_extraction_result)
    monkeypatch.setattr(_metadata_mod, "_extract_pdf_metadata", mock_extract_metadata)

    mock_file = MagicMock()
    mock_file.filename = "contract.pdf"
    mock_file.content_type = "application/pdf"
    mock_file.seek = AsyncMock()
    pdf_dummy_content = b"%PDF-1.4 dummy"
    mock_file.read = AsyncMock(return_value=pdf_dummy_content)

    outcome: StageOutcome = await stage_metadata(mock_file)

    # Check that _extract_pdf_metadata was called with content and filename
    mock_extract_metadata.assert_called_once_with(pdf_dummy_content, "contract.pdf")
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
    """Any unexpected error inside _extract_pdf_metadata should be wrapped and raised as MetadataProcessingError."""

    # Force the helper to raise to exercise the exception branch
    async def _boom_mock(
        content_bytes: bytes, filename: str | None
    ) -> str:  # pragma: no cover
        raise RuntimeError("explode")

    monkeypatch.setattr(_metadata_mod, "_extract_pdf_metadata", _boom_mock)

    mock_file = MagicMock()
    mock_file.filename = "bad.pdf"
    mock_file.content_type = "application/pdf"
    mock_file.seek = AsyncMock()
    mock_file.read = AsyncMock(return_value=b"%PDF bad")

    with pytest.raises(MetadataProcessingError) as excinfo:
        await stage_metadata(mock_file)

    # The original "explode" should be part of the raised MetadataProcessingError's message
    # due to the wrapping `raise MetadataProcessingError(...) from e`
    assert "General processing error in metadata stage: explode" in str(excinfo.value)


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
    # Replacement character (0xFFFD) will appear where invalid byte was located
    assert "bad" in text and "data" in text
