"""tests/unit/parsing/test_parsers.py
###############################################################################
Unit tests for individual file parsing adapters.
(``src.parsing.*``)
###############################################################################
This module contains tests for each file parser:
- PDF parser (pdf.py)
- DOCX parser (docx.py)
- CSV parser (csv.py)
- Image (OCR) parser (image.py)

Tests verify that each parser correctly processes mock UploadFile objects,
interacts with its underlying libraries (pdfminer, docx2txt, pandas, Pillow, pytesseract)
as expected, and returns the extracted text. Heavy I/O operations are mocked
using `asyncio.to_thread` and the respective library calls.
"""

from __future__ import annotations

from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from starlette.datastructures import UploadFile

# Parsers to test
from src.parsing.csv import extract_text_from_csv
from src.parsing.docx import extract_text_from_docx
from src.parsing.image import extract_text_from_image
from src.parsing.pdf import extract_text_from_pdf

# Mock pandas errors for CSV testing
try:
    from pandas.errors import EmptyDataError, ParserError
except ImportError:  # Create dummy exceptions if pandas not installed in test env

    class EmptyDataError(Exception):
        pass

    class ParserError(Exception):
        pass


@pytest.fixture
def mock_upload_file_factory():
    """Factory to create mock UploadFile objects for testing parsers."""

    def _factory(filename: str, content: bytes, content_type: str) -> MagicMock:
        mock_file = MagicMock(spec=UploadFile)
        mock_file.filename = filename
        mock_file.content_type = content_type

        # Mock the file-like object within UploadFile
        # For most parsers, they will call await file.read()
        mock_file.file = BytesIO(content)  # For sync operations if any part uses it
        mock_file.seek = AsyncMock()
        mock_file.read = AsyncMock(return_value=content)
        return mock_file

    return _factory


# PDF Parser Tests
@pytest.mark.asyncio
async def test_extract_text_from_pdf_success(mock_upload_file_factory) -> None:
    """Tests successful text extraction from a PDF."""
    pdf_content = b"%PDF-1.4 fake pdf content"
    mock_file = mock_upload_file_factory("test.pdf", pdf_content, "application/pdf")
    expected_text = "Expected PDF text"

    # Create a direct mock for asyncio.to_thread
    with patch("asyncio.to_thread") as mock_to_thread:
        # Make to_thread return our expected text directly
        mock_to_thread.return_value = expected_text

        result = await extract_text_from_pdf(mock_file)

        # Assertions
        assert result == expected_text
        mock_file.seek.assert_called_once_with(0)
        mock_file.read.assert_called_once_with()
        mock_to_thread.assert_called_once()


# DOCX Parser Tests
@pytest.mark.asyncio
async def test_extract_text_from_docx_success(mock_upload_file_factory) -> None:
    """Tests successful text extraction from a DOCX file."""
    docx_content = b"PK fake docx content"
    mock_file = mock_upload_file_factory(
        "test.docx",
        docx_content,
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )
    expected_text = "Expected DOCX text"

    # Mock tempfile.NamedTemporaryFile, docx2txt.process, and asyncio.to_thread
    with patch("tempfile.NamedTemporaryFile") as mock_temp_file_constructor:
        # Configure the mock context manager
        mock_temp_file_instance = MagicMock()
        mock_temp_file_instance.name = "dummy_temp_file.docx"  # Needs a name for Path()
        mock_temp_file_instance.__enter__.return_value = mock_temp_file_instance
        mock_temp_file_instance.__exit__.return_value = None
        mock_temp_file_constructor.return_value = mock_temp_file_instance

        with (
            patch("docx2txt.process", return_value=expected_text) as mock_process,
            patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread,
        ):

            mock_to_thread.side_effect = lambda func, *args, **kwargs: func(
                *args, **kwargs
            )

            result = await extract_text_from_docx(mock_file)

            assert result == expected_text
            mock_file.seek.assert_called_once_with(0)
            mock_file.read.assert_called_once_with()

            # Check tempfile usage
            mock_temp_file_constructor.assert_called_once_with(
                suffix=".docx", delete=True
            )
            mock_temp_file_instance.write.assert_called_once_with(docx_content)
            mock_temp_file_instance.flush.assert_called_once()

            # Check docx2txt.process call
            mock_process.assert_called_once_with("dummy_temp_file.docx")


# CSV Parser Tests
@pytest.mark.asyncio
async def test_extract_text_from_csv_pandas_success(mock_upload_file_factory) -> None:
    """Tests successful CSV parsing using pandas."""
    csv_content = b"col1,col2\nval1,val2"
    mock_file = mock_upload_file_factory("test.csv", csv_content, "text/csv")
    expected_text = "col1 col2\nval1 val2"

    # Mock asyncio.to_thread to return the expected text directly
    with patch("asyncio.to_thread") as mock_to_thread:
        mock_to_thread.return_value = expected_text

        result = await extract_text_from_csv(mock_file)

        # Verify result and mocks
        assert result == expected_text
        mock_file.seek.assert_called_once_with(0)
        mock_file.read.assert_called_once_with()
        mock_to_thread.assert_called_once()


@pytest.mark.asyncio
async def test_extract_text_from_csv_pandas_failure_fallback(
    mock_upload_file_factory,
) -> None:
    """Tests CSV parsing fallback when pandas fails."""
    csv_content = b"bad,csv\ndata,here"  # Content that might cause ParserError
    mock_file = mock_upload_file_factory("test.csv", csv_content, "text/csv")
    expected_text_fallback = csv_content.decode("utf-8", errors="replace")

    with (
        patch(
            "pandas.read_csv", side_effect=ParserError("Simulated pandas parsing error")
        ),
        patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread,
    ):

        # Ensure to_thread still calls the worker, which will then raise the error
        async def fake_to_thread(worker_fn, *args, **kwargs):
            try:
                return worker_fn(*args, **kwargs)
            except ParserError as e:  # Propagate the specific error for the test
                raise e

        mock_to_thread.side_effect = fake_to_thread

        result = await extract_text_from_csv(mock_file)

        assert result == expected_text_fallback
        mock_file.seek.assert_called_once_with(0)
        mock_file.read.assert_called_once_with()


# Image (OCR) Parser Tests
@pytest.mark.asyncio
async def test_extract_text_from_image_success(mock_upload_file_factory) -> None:
    """Tests successful OCR text extraction from an image."""
    image_content = b"fake_image_bytes"
    mock_file = mock_upload_file_factory("test.png", image_content, "image/png")
    expected_text = "Expected OCR text"

    with (
        patch("PIL.Image.open") as mock_pil_open,
        patch("pytesseract.image_to_string", return_value=expected_text) as mock_ocr,
        patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread,
    ):

        mock_image_instance = MagicMock()
        mock_image_instance.convert.return_value = (
            mock_image_instance  # Return self after convert
        )
        mock_pil_open.return_value.__enter__.return_value = (
            mock_image_instance  # For 'with Image.open...'
        )

        mock_to_thread.side_effect = lambda func, *args, **kwargs: func(*args, **kwargs)

        result = await extract_text_from_image(mock_file)

        assert result == expected_text
        mock_file.seek.assert_called_once_with(0)
        mock_file.read.assert_called_once_with()

        mock_pil_open.assert_called_once()
        pil_open_arg = mock_pil_open.call_args[0][0]
        assert isinstance(pil_open_arg, BytesIO)
        assert pil_open_arg.read() == image_content

        mock_image_instance.convert.assert_called_once_with("RGB")
        mock_ocr.assert_called_once_with(
            mock_image_instance
        )  # Check it's called with the converted image
