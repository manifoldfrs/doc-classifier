"""tests/unit/classification/stages/test_common_stages.py
###############################################################################
Unit tests for individual classification stages.
(``src.classification.stages.*``)
###############################################################################
This module contains tests for each classification stage:
- Filename stage
- Metadata stage
- Text stage
- OCR stage

Tests verify that each stage correctly processes mock UploadFile objects,
interacts with its dependencies (like parsers or ML models) as expected,
and returns the correct StageOutcome.
"""

from __future__ import annotations

from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from starlette.datastructures import UploadFile

# Stages to test
from src.classification.stages.filename import stage_filename
from src.classification.stages.metadata import stage_metadata
from src.classification.stages.ocr import stage_ocr
from src.classification.stages.text import stage_text


@pytest.fixture
def mock_upload_file_factory():
    """Factory to create mock UploadFile objects for testing stages."""

    def _factory(filename: str, content: bytes, content_type: str) -> MagicMock:
        mock_file = MagicMock(spec=UploadFile)
        mock_file.filename = filename
        mock_file.content_type = content_type

        # Mock the file-like object within UploadFile
        mock_file.file = BytesIO(content)  # Use BytesIO for seek/read
        # For stages that might use async seek/read on UploadFile itself
        mock_file.seek = AsyncMock()
        mock_file.read = AsyncMock(return_value=content)
        return mock_file

    return _factory


# Test Filename Stage
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "filename, expected_label, expected_confidence_range",
    [
        ("invoice_123.pdf", "invoice", (0.80, 0.95)),
        ("my_bank_statement.docx", "bank_statement", (0.80, 0.95)),
        ("financial_report_final.xlsx", "financial_report", (0.80, 0.95)),
        ("drivers_license_scan.jpg", "drivers_licence", (0.80, 0.95)),
        ("id_card_john_doe.png", "id_doc", (0.80, 0.95)),
        ("service_agreement.pdf", "contract", (0.80, 0.95)),
        ("important_email.eml", "email", (0.80, 0.95)),  # .eml specific check
        ("application_form_v2.pdf", "form", (0.80, 0.95)),
        ("unknown_document.dat", None, None),
        ("", None, None),  # Empty filename
        ("path/to/invoice.pdf", "invoice", (0.80, 0.95)),  # With path
        ("INV001.pdf", "invoice", (0.80, 0.95)),  # Strong start
    ],
)
async def test_stage_filename(
    filename: str,
    expected_label: str | None,
    expected_confidence_range: tuple[float, float] | None,
    mock_upload_file_factory,
) -> None:
    """Tests the filename stage with various inputs."""
    mock_file = mock_upload_file_factory(filename, b"dummy", "application/octet-stream")
    outcome = await stage_filename(mock_file)

    assert outcome.label == expected_label
    if expected_confidence_range and outcome.confidence is not None:
        assert (
            expected_confidence_range[0]
            <= outcome.confidence
            <= expected_confidence_range[1]
        )
    else:
        assert outcome.confidence is None


# Test Metadata Stage
@pytest.mark.asyncio
async def test_stage_metadata_pdf_match(mock_upload_file_factory) -> None:
    """Tests metadata stage with a PDF that has matching metadata."""
    mock_file = mock_upload_file_factory(
        "meta_invoice.pdf", b"pdf_content", "application/pdf"
    )

    with patch(
        "src.classification.stages.metadata._extract_pdf_metadata",
        AsyncMock(return_value="This is an Invoice"),
    ) as mock_extract:
        outcome = await stage_metadata(mock_file)
        mock_extract.assert_called_once_with(b"pdf_content")
        assert outcome.label == "invoice"
        assert outcome.confidence == pytest.approx(0.86)


@pytest.mark.asyncio
async def test_stage_metadata_pdf_no_match(mock_upload_file_factory) -> None:
    """Tests metadata stage with a PDF that has no matching metadata."""
    mock_file = mock_upload_file_factory(
        "other_doc.pdf", b"pdf_content", "application/pdf"
    )
    with patch(
        "src.classification.stages.metadata._extract_pdf_metadata",
        AsyncMock(return_value="Generic document info"),
    ) as mock_extract:
        outcome = await stage_metadata(mock_file)
        mock_extract.assert_called_once_with(b"pdf_content")
        assert outcome.label is None
        assert outcome.confidence is None


@pytest.mark.asyncio
async def test_stage_metadata_not_pdf(mock_upload_file_factory) -> None:
    """Tests metadata stage with a non-PDF file, should skip."""
    mock_file = mock_upload_file_factory("document.txt", b"text_content", "text/plain")
    with patch(
        "src.classification.stages.metadata._extract_pdf_metadata",
        new_callable=AsyncMock,
    ) as mock_extract:
        outcome = await stage_metadata(mock_file)
        mock_extract.assert_not_called()
        assert outcome.label is None
        assert outcome.confidence is None


@pytest.mark.asyncio
async def test_stage_metadata_pdf_extraction_fails(mock_upload_file_factory) -> None:
    """Tests metadata stage when PDF metadata extraction returns empty string (simulating failure)."""
    mock_file = mock_upload_file_factory("corrupt.pdf", b"bad_pdf", "application/pdf")
    with patch(
        "src.classification.stages.metadata._extract_pdf_metadata",
        AsyncMock(return_value=""),
    ) as mock_extract:
        outcome = await stage_metadata(mock_file)
        mock_extract.assert_called_once_with(b"bad_pdf")
        assert outcome.label is None
        assert outcome.confidence is None


# Test Text Stage
@pytest.mark.asyncio
async def test_stage_text_with_model(mock_upload_file_factory) -> None:
    """Tests text stage when ML model is available and predicts."""
    mock_file = mock_upload_file_factory("invoice.pdf", b"content", "application/pdf")

    # Mock the TEXT_EXTRACTORS for 'pdf'
    mock_pdf_parser = AsyncMock(return_value="extracted invoice text")

    with (
        patch(
            "src.classification.stages.text.TEXT_EXTRACTORS", {"pdf": mock_pdf_parser}
        ),
        patch("src.classification.stages.text._MODEL_AVAILABLE", True),
        patch(
            "src.classification.stages.text._model.predict",
            return_value=("invoice_model", 0.88),
        ) as mock_model_predict,
    ):

        outcome = await stage_text(mock_file)

        mock_pdf_parser.assert_called_once_with(mock_file)
        mock_model_predict.assert_called_once_with("extracted invoice text")
        assert outcome.label == "invoice_model"
        assert outcome.confidence == pytest.approx(0.88)


@pytest.mark.asyncio
async def test_stage_text_model_unavailable_fallback_heuristic(
    mock_upload_file_factory,
) -> None:
    """Tests text stage fallback to heuristics when model is unavailable."""
    mock_file = mock_upload_file_factory("statement.csv", b"content", "text/csv")
    mock_csv_parser = AsyncMock(return_value="bank statement keywords here")

    with (
        patch(
            "src.classification.stages.text.TEXT_EXTRACTORS", {"csv": mock_csv_parser}
        ),
        patch("src.classification.stages.text._MODEL_AVAILABLE", False),
    ):  # Simulate model not available

        outcome = await stage_text(mock_file)

        mock_csv_parser.assert_called_once_with(mock_file)
        assert outcome.label == "bank_statement"  # From heuristic
        assert outcome.confidence == pytest.approx(0.75)  # Fallback confidence


@pytest.mark.asyncio
async def test_stage_text_unsupported_extension(mock_upload_file_factory) -> None:
    """Tests text stage with an unsupported text file extension."""
    mock_file = mock_upload_file_factory("archive.zip", b"content", "application/zip")
    # TEXT_EXTRACTORS won't have "zip"
    with patch("src.classification.stages.text.TEXT_EXTRACTORS", {}):
        outcome = await stage_text(mock_file)
        assert outcome.label is None
        assert outcome.confidence is None


@pytest.mark.asyncio
async def test_stage_text_empty_extracted_text(mock_upload_file_factory) -> None:
    """Tests text stage when the parser returns empty text."""
    mock_file = mock_upload_file_factory("empty.txt", b"", "text/plain")
    mock_txt_parser = AsyncMock(return_value="  ")  # Whitespace only

    with (
        patch(
            "src.classification.stages.text.TEXT_EXTRACTORS", {"txt": mock_txt_parser}
        ),
        patch("src.classification.stages.text._MODEL_AVAILABLE", False),
    ):
        outcome = await stage_text(mock_file)
        assert outcome.label is None
        assert outcome.confidence is None


# Test OCR Stage
@pytest.mark.asyncio
async def test_stage_ocr_with_model(mock_upload_file_factory) -> None:
    """Tests OCR stage when ML model is available and predicts."""
    mock_file = mock_upload_file_factory("license.png", b"img_content", "image/png")
    mock_image_parser = AsyncMock(return_value="ocr text drivers license")

    with (
        patch(
            "src.classification.stages.ocr.IMAGE_EXTRACTORS", {"png": mock_image_parser}
        ),
        patch("src.classification.stages.ocr._MODEL_AVAILABLE", True),
        patch(
            "src.classification.stages.ocr._model.predict",
            return_value=("drivers_licence_model", 0.91),
        ) as mock_model_predict,
    ):

        outcome = await stage_ocr(mock_file)

        mock_image_parser.assert_called_once_with(mock_file)
        mock_model_predict.assert_called_once_with("ocr text drivers license")
        assert outcome.label == "drivers_licence_model"
        assert outcome.confidence == pytest.approx(0.91)


@pytest.mark.asyncio
async def test_stage_ocr_model_unavailable_fallback_heuristic(
    mock_upload_file_factory,
) -> None:
    """Tests OCR stage fallback to heuristics when model is unavailable."""
    mock_file = mock_upload_file_factory("photo_id.jpg", b"img_content", "image/jpeg")
    mock_image_parser = AsyncMock(return_value="some form application text")

    with (
        patch(
            "src.classification.stages.ocr.IMAGE_EXTRACTORS", {"jpg": mock_image_parser}
        ),
        patch("src.classification.stages.ocr._MODEL_AVAILABLE", False),
    ):  # Simulate model not available

        outcome = await stage_ocr(mock_file)

        mock_image_parser.assert_called_once_with(mock_file)
        assert outcome.label == "form"  # From heuristic
        assert outcome.confidence == pytest.approx(0.72)  # OCR Fallback confidence


@pytest.mark.asyncio
async def test_stage_ocr_unsupported_extension(mock_upload_file_factory) -> None:
    """Tests OCR stage with an unsupported image file extension."""
    mock_file = mock_upload_file_factory(
        "document.pdf", b"pdf_content", "application/pdf"
    )
    # IMAGE_EXTRACTORS won't have "pdf"
    with patch("src.classification.stages.ocr.IMAGE_EXTRACTORS", {}):
        outcome = await stage_ocr(mock_file)
        assert outcome.label is None
        assert outcome.confidence is None


@pytest.mark.asyncio
async def test_stage_ocr_empty_extracted_text(mock_upload_file_factory) -> None:
    """Tests OCR stage when the image parser (OCR) returns empty text."""
    mock_file = mock_upload_file_factory("blank_image.png", b"img_content", "image/png")
    mock_image_parser = AsyncMock(return_value="\n \t ")  # Whitespace only

    with (
        patch(
            "src.classification.stages.ocr.IMAGE_EXTRACTORS", {"png": mock_image_parser}
        ),
        patch("src.classification.stages.ocr._MODEL_AVAILABLE", False),
    ):
        outcome = await stage_ocr(mock_file)
        assert outcome.label is None
        assert outcome.confidence is None
