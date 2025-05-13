from __future__ import annotations

from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from starlette.datastructures import UploadFile

from src.classification.model import ModelNotAvailableError
from src.classification.stages.filename import stage_filename
from src.classification.stages.metadata import stage_metadata
from src.classification.stages.ocr import stage_ocr
from src.classification.stages.text import stage_text
from src.classification.types import StageOutcome


@pytest.fixture
def mock_upload_file_factory():
    """Factory to create mock UploadFile objects for testing stages."""

    def _factory(
        filename: str, content: bytes, content_type: str | None = None
    ) -> MagicMock:
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
        (None, None, None),  # None filename
        ("path/to/invoice.pdf", "invoice", (0.80, 0.95)),  # With path
        ("INV001.pdf", "invoice", (0.80, 0.95)),  # Strong start
    ],
)
async def test_stage_filename(
    filename: str | None,  # Allow None
    expected_label: str | None,
    expected_confidence_range: tuple[float, float] | None,
    mock_upload_file_factory,
) -> None:
    """Tests the filename stage with various inputs."""
    # Handle None filename case for factory
    if filename is None:
        mock_file = mock_upload_file_factory(
            "dummy", b"dummy", "application/octet-stream"
        )
        mock_file.filename = None  # Explicitly set to None after creation
    else:
        mock_file = mock_upload_file_factory(
            filename, b"dummy", "application/octet-stream"
        )

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


@pytest.mark.asyncio
async def test_stage_metadata_pdf_extraction_raises_exception(
    mock_upload_file_factory,
) -> None:
    """Tests metadata stage when _extract_pdf_metadata raises an unexpected exception."""
    mock_file = mock_upload_file_factory("error.pdf", b"pdf_content", "application/pdf")
    # Simulate an error during the extraction process within the stage
    with (
        patch(
            "src.classification.stages.metadata._extract_pdf_metadata",
            AsyncMock(side_effect=Exception("Unexpected PDF processing error")),
        ),
        patch("src.classification.stages.metadata.logger") as mock_logger,
    ):
        outcome = await stage_metadata(mock_file)

        # Assert that the stage returns None/None outcome upon error
        assert outcome == StageOutcome(label=None, confidence=None)
        # Check that the error was logged
        mock_logger.error.assert_called_once()
        assert "metadata_stage_processing_error" in mock_logger.error.call_args[0]
        assert (
            "Unexpected PDF processing error" in mock_logger.error.call_args[1]["error"]
        )


@pytest.mark.asyncio
async def test_stage_metadata_file_read_raises_exception(
    mock_upload_file_factory,
) -> None:
    """Tests metadata stage when file.read() raises an exception."""
    mock_file = mock_upload_file_factory("io_error.pdf", b"", "application/pdf")
    # Simulate an error during file reading
    mock_file.read.side_effect = OSError("Simulated read error")

    with patch("src.classification.stages.metadata.logger") as mock_logger:
        outcome = await stage_metadata(mock_file)

        assert outcome == StageOutcome(label=None, confidence=None)
        mock_logger.error.assert_called_once()
        assert "metadata_stage_processing_error" in mock_logger.error.call_args[0]
        assert "Simulated read error" in mock_logger.error.call_args[1]["error"]


# Test Text Stage
@pytest.mark.asyncio
async def test_stage_text_with_model(mock_upload_file_factory) -> None:
    """Tests text stage when ML model is available and predicts."""
    mock_file = mock_upload_file_factory("invoice.pdf", b"content", "application/pdf")
    mock_pdf_parser = AsyncMock(return_value="extracted invoice text")

    # Patch the TEXT_EXTRACTORS within the text stage module
    # Patch the imported 'predict' function within the text stage module
    with (
        patch.dict(
            "src.classification.stages.text.TEXT_EXTRACTORS", {"pdf": mock_pdf_parser}
        ),
        patch("src.classification.stages.text._MODEL_AVAILABLE", True),
        patch(
            "src.classification.stages.text.predict",
            return_value=("invoice_model", 0.88),
        ) as mock_model_predict,
        patch("src.classification.stages.text.logger") as mock_logger,
    ):
        outcome = await stage_text(mock_file)

        mock_file.seek.assert_called_once_with(0)
        mock_pdf_parser.assert_called_once_with(mock_file)
        mock_model_predict.assert_called_once_with("extracted invoice text")
        assert outcome.label == "invoice_model"
        assert outcome.confidence == pytest.approx(0.88)
        mock_logger.debug.assert_any_call(
            "text_stage_model_prediction",
            filename="invoice.pdf",
            label="invoice_model",
            confidence=0.88,
        )


@pytest.mark.asyncio
async def test_stage_text_model_unavailable_fallback_heuristic(
    mock_upload_file_factory,
) -> None:
    """Tests text stage fallback to heuristics when model is unavailable."""
    mock_file = mock_upload_file_factory("statement.csv", b"content", "text/csv")
    mock_csv_parser = AsyncMock(return_value="bank statement keywords here")

    # Patch 'predict' to raise ModelNotAvailableError
    with (
        patch.dict(
            "src.classification.stages.text.TEXT_EXTRACTORS", {"csv": mock_csv_parser}
        ),
        patch(
            "src.classification.stages.text._MODEL_AVAILABLE", True
        ),  # Model is configured to be available
        patch(
            "src.classification.stages.text.predict",
            side_effect=ModelNotAvailableError("Model not found"),
        ) as mock_model_predict,
        patch("src.classification.stages.text.logger") as mock_logger,
    ):
        outcome = await stage_text(mock_file)

        mock_file.seek.assert_called_once_with(0)
        mock_csv_parser.assert_called_once_with(mock_file)
        mock_model_predict.assert_called_once_with(
            "bank statement keywords here"
        )  # Check predict was called
        mock_logger.warning.assert_called_once_with(
            "text_stage_model_not_available",
            filename="statement.csv",
            fallback="heuristics",
        )
        mock_logger.debug.assert_any_call(  # Check heuristic match logging
            "text_stage_heuristic_match",
            filename="statement.csv",
            label="bank_statement",
            confidence=0.75,
        )
        assert outcome.label == "bank_statement"  # From heuristic
        assert outcome.confidence == pytest.approx(0.75)  # Fallback confidence


@pytest.mark.asyncio
async def test_stage_text_model_prediction_error(mock_upload_file_factory) -> None:
    """Tests text stage when model prediction itself raises an error."""
    mock_file = mock_upload_file_factory("error.txt", b"content", "text/plain")
    mock_txt_parser = AsyncMock(return_value="some text content")

    with (
        patch.dict(
            "src.classification.stages.text.TEXT_EXTRACTORS", {"txt": mock_txt_parser}
        ),
        patch("src.classification.stages.text._MODEL_AVAILABLE", True),
        patch(
            "src.classification.stages.text.predict",
            side_effect=Exception("ML model runtime error"),
        ) as mock_model_predict,
        patch("src.classification.stages.text.logger") as mock_logger,
    ):
        outcome = await stage_text(mock_file)

        mock_model_predict.assert_called_once_with("some text content")
        # Stage should return None/None upon prediction error
        assert outcome == StageOutcome(label=None, confidence=None)
        # Check that the error was logged
        mock_logger.error.assert_called_once()
        assert "text_stage_model_prediction_error" in mock_logger.error.call_args[0]
        assert "ML model runtime error" in mock_logger.error.call_args[1]["error"]


@pytest.mark.asyncio
async def test_stage_text_extractor_error(mock_upload_file_factory) -> None:
    """Tests text stage when the text extractor itself fails."""
    mock_file = mock_upload_file_factory(
        "extract_fail.pdf", b"content", "application/pdf"
    )
    # Make the mock extractor raise an error
    mock_pdf_parser = AsyncMock(side_effect=IOError("Cannot read PDF"))

    with (
        patch.dict(
            "src.classification.stages.text.TEXT_EXTRACTORS", {"pdf": mock_pdf_parser}
        ),
        patch("src.classification.stages.text.logger") as mock_logger,
    ):
        outcome = await stage_text(mock_file)

        # Expect None/None outcome
        assert outcome == StageOutcome(label=None, confidence=None)
        # Check that the extraction error was logged
        mock_logger.error.assert_called_once()
        assert "text_stage_extraction_error" in mock_logger.error.call_args[0]
        assert "Cannot read PDF" in mock_logger.error.call_args[1]["error"]


@pytest.mark.asyncio
async def test_stage_text_unsupported_extension(mock_upload_file_factory) -> None:
    """Tests text stage with an unsupported text file extension."""
    mock_file = mock_upload_file_factory("archive.zip", b"content", "application/zip")
    # Ensure TEXT_EXTRACTORS doesn't have 'zip' by patching it (or ensure default doesn't)
    with patch.dict("src.classification.stages.text.TEXT_EXTRACTORS", {}, clear=True):
        outcome = await stage_text(mock_file)
        assert outcome.label is None
        assert outcome.confidence is None


@pytest.mark.asyncio
async def test_stage_text_empty_extracted_text(mock_upload_file_factory) -> None:
    """Tests text stage when the parser returns empty text."""
    mock_file = mock_upload_file_factory("empty.txt", b"", "text/plain")
    mock_txt_parser = AsyncMock(return_value="  ")  # Whitespace only

    with patch.dict(
        "src.classification.stages.text.TEXT_EXTRACTORS", {"txt": mock_txt_parser}
    ):
        outcome = await stage_text(mock_file)
        mock_file.seek.assert_called_once_with(0)
        mock_txt_parser.assert_called_once_with(mock_file)
        assert outcome.label is None
        assert outcome.confidence is None


# Test OCR Stage
@pytest.mark.asyncio
async def test_stage_ocr_with_model(mock_upload_file_factory) -> None:
    """Tests OCR stage when ML model is available and predicts."""
    mock_file = mock_upload_file_factory("license.png", b"img_content", "image/png")
    mock_image_parser = AsyncMock(return_value="ocr text drivers license")

    # Patch the IMAGE_EXTRACTORS within the ocr stage module
    # Patch the imported 'predict' function within the ocr stage module
    with (
        patch.dict(
            "src.classification.stages.ocr.IMAGE_EXTRACTORS", {"png": mock_image_parser}
        ),
        patch("src.classification.stages.ocr._MODEL_AVAILABLE", True),
        patch(
            "src.classification.stages.ocr.predict",
            return_value=("drivers_licence_model", 0.91),
        ) as mock_model_predict,
        patch("src.classification.stages.ocr.logger") as mock_logger,
    ):
        outcome = await stage_ocr(mock_file)

        mock_file.seek.assert_called_once_with(0)
        mock_image_parser.assert_called_once_with(mock_file)
        mock_model_predict.assert_called_once_with("ocr text drivers license")
        assert outcome.label == "drivers_licence_model"
        assert outcome.confidence == pytest.approx(0.91)
        mock_logger.debug.assert_any_call(
            "ocr_stage_model_prediction",
            filename="license.png",
            label="drivers_licence_model",
            confidence=0.91,
        )


@pytest.mark.asyncio
async def test_stage_ocr_model_unavailable_fallback_heuristic(
    mock_upload_file_factory,
) -> None:
    """Tests OCR stage fallback to heuristics when model is unavailable."""
    mock_file = mock_upload_file_factory("photo_id.jpg", b"img_content", "image/jpeg")
    mock_image_parser = AsyncMock(return_value="some form application text")

    # Patch 'predict' to raise ModelNotAvailableError
    with (
        patch.dict(
            "src.classification.stages.ocr.IMAGE_EXTRACTORS", {"jpg": mock_image_parser}
        ),
        patch(
            "src.classification.stages.ocr._MODEL_AVAILABLE", True
        ),  # Model is available
        patch(
            "src.classification.stages.ocr.predict",
            side_effect=ModelNotAvailableError("Model not found"),
        ) as mock_model_predict,
        patch("src.classification.stages.ocr.logger") as mock_logger,
    ):
        outcome = await stage_ocr(mock_file)

        mock_file.seek.assert_called_once_with(0)
        mock_image_parser.assert_called_once_with(mock_file)
        mock_model_predict.assert_called_once_with(
            "some form application text"
        )  # Check predict was called
        mock_logger.warning.assert_called_once_with(
            "ocr_stage_model_not_available",
            filename="photo_id.jpg",
            fallback="heuristics",
        )
        mock_logger.debug.assert_any_call(  # Check heuristic match logging
            "ocr_stage_heuristic_match",
            filename="photo_id.jpg",
            label="form",
            confidence=0.72,
        )
        assert outcome.label == "form"  # From heuristic
        assert outcome.confidence == pytest.approx(0.72)  # Fallback confidence


@pytest.mark.asyncio
async def test_stage_ocr_model_prediction_error(mock_upload_file_factory) -> None:
    """Tests OCR stage when model prediction itself raises an error."""
    mock_file = mock_upload_file_factory("error.png", b"img_content", "image/png")
    mock_image_parser = AsyncMock(return_value="some ocr text")

    with (
        patch.dict(
            "src.classification.stages.ocr.IMAGE_EXTRACTORS", {"png": mock_image_parser}
        ),
        patch("src.classification.stages.ocr._MODEL_AVAILABLE", True),
        patch(
            "src.classification.stages.ocr.predict",
            side_effect=Exception("ML model runtime error"),
        ) as mock_model_predict,
        patch("src.classification.stages.ocr.logger") as mock_logger,
    ):
        outcome = await stage_ocr(mock_file)

        mock_model_predict.assert_called_once_with("some ocr text")
        # Stage should return None/None upon prediction error
        assert outcome == StageOutcome(label=None, confidence=None)
        # Check that the error was logged
        mock_logger.error.assert_called_once()
        assert "ocr_stage_model_prediction_error" in mock_logger.error.call_args[0]
        assert "ML model runtime error" in mock_logger.error.call_args[1]["error"]


@pytest.mark.asyncio
async def test_stage_ocr_extractor_error(mock_upload_file_factory) -> None:
    """Tests OCR stage when the image extractor (OCR) itself fails."""
    mock_file = mock_upload_file_factory(
        "extract_fail.jpg", b"img_content", "image/jpeg"
    )
    # Make the mock extractor raise an error
    mock_image_parser = AsyncMock(side_effect=RuntimeError("OCR engine failed"))

    with (
        patch.dict(
            "src.classification.stages.ocr.IMAGE_EXTRACTORS", {"jpg": mock_image_parser}
        ),
        patch("src.classification.stages.ocr.logger") as mock_logger,
    ):
        outcome = await stage_ocr(mock_file)

        # Expect None/None outcome
        assert outcome == StageOutcome(label=None, confidence=None)
        # Check that the extraction error was logged
        mock_logger.error.assert_called_once()
        assert "ocr_stage_extraction_error" in mock_logger.error.call_args[0]
        assert "OCR engine failed" in mock_logger.error.call_args[1]["error"]


@pytest.mark.asyncio
async def test_stage_ocr_unsupported_extension(mock_upload_file_factory) -> None:
    """Tests OCR stage with an unsupported image file extension."""
    mock_file = mock_upload_file_factory(
        "document.pdf", b"pdf_content", "application/pdf"
    )
    # Ensure IMAGE_EXTRACTORS doesn't have 'pdf'
    with patch.dict("src.classification.stages.ocr.IMAGE_EXTRACTORS", {}, clear=True):
        outcome = await stage_ocr(mock_file)
        assert outcome.label is None
        assert outcome.confidence is None


@pytest.mark.asyncio
async def test_stage_ocr_empty_extracted_text(mock_upload_file_factory) -> None:
    """Tests OCR stage when the image parser (OCR) returns empty text."""
    mock_file = mock_upload_file_factory("blank_image.png", b"img_content", "image/png")
    mock_image_parser = AsyncMock(return_value="\n \t ")  # Whitespace only

    with patch.dict(
        "src.classification.stages.ocr.IMAGE_EXTRACTORS", {"png": mock_image_parser}
    ):
        outcome = await stage_ocr(mock_file)
        mock_file.seek.assert_called_once_with(0)
        mock_image_parser.assert_called_once_with(mock_file)
        assert outcome.label is None
        assert outcome.confidence is None
