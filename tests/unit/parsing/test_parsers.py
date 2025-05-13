from __future__ import annotations

from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from starlette.datastructures import UploadFile

# Parsers to test
from src.parsing.csv import extract_text_from_csv
from src.parsing.docx import extract_text_from_docx
from src.parsing.image import extract_text_from_image
from src.parsing.pdf import PDFException, extract_text_from_pdf
from src.parsing.txt import read_txt

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

    def _factory(
        filename: str, content: bytes, content_type: str | None = None
    ) -> MagicMock:
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
        # Verify the worker function passed to to_thread exists
        assert callable(mock_to_thread.call_args[0][0])
        # Verify the pdf content bytes are passed to the worker
        assert mock_to_thread.call_args[0][1] == pdf_content


@pytest.mark.asyncio
async def test_extract_text_from_pdf_extraction_error(mock_upload_file_factory) -> None:
    """Tests handling of pdfminer extraction errors."""
    pdf_content = b"%PDF-corrupted"
    mock_file = mock_upload_file_factory("corrupt.pdf", pdf_content, "application/pdf")

    # Mock extract_text to raise an exception inside the worker
    with patch(
        "src.parsing.pdf.extract_text", side_effect=PDFException("Simulated PDF error")
    ):
        # Use asyncio.to_thread wrapper like the actual code
        with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread:
            # Make to_thread execute the actual worker which will raise the error
            mock_to_thread.side_effect = lambda func, *args, **kwargs: func(
                *args, **kwargs
            )

            result = await extract_text_from_pdf(mock_file)

            # Assertions
            assert result == ""  # Should return empty string on error
            mock_file.seek.assert_called_once_with(0)
            mock_file.read.assert_called_once_with()


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


@pytest.mark.asyncio
async def test_extract_text_from_docx_extraction_error(
    mock_upload_file_factory,
) -> None:
    """Tests handling of errors during docx2txt processing."""
    docx_content = b"PK corrupted docx"
    mock_file = mock_upload_file_factory(
        "bad.docx",
        docx_content,
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )

    with patch("tempfile.NamedTemporaryFile") as mock_temp_file_constructor:
        mock_temp_file_instance = MagicMock()
        mock_temp_file_instance.name = "dummy_temp_file.docx"
        mock_temp_file_instance.__enter__.return_value = mock_temp_file_instance
        mock_temp_file_instance.__exit__.return_value = None
        mock_temp_file_constructor.return_value = mock_temp_file_instance

        with (
            patch(
                "docx2txt.process", side_effect=Exception("Simulated docx error")
            ) as mock_process,
            patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread,
        ):
            mock_to_thread.side_effect = lambda func, *args, **kwargs: func(
                *args, **kwargs
            )

            result = await extract_text_from_docx(mock_file)

            # Expecting an error message string
            assert "Error extracting DOCX text: Simulated docx error" in result
            mock_process.assert_called_once()


# CSV Parser Tests
@pytest.mark.asyncio
async def test_extract_text_from_csv_pandas_success(mock_upload_file_factory) -> None:
    """Tests successful CSV parsing using pandas."""
    csv_content = b"col1,col2\nval1,val2\nval3,val4"  # Added another row
    mock_file = mock_upload_file_factory("test.csv", csv_content, "text/csv")
    expected_text = "col1 col2\nval1 val2\nval3 val4"  # Updated expected

    # Mock pandas.read_csv within the worker's scope
    with (
        patch("pandas.read_csv") as mock_read_csv,
        patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread,
    ):
        # Configure mock DataFrame
        import pandas as pd  # Import locally for DataFrame creation

        mock_df = pd.DataFrame({"col1": ["val1", "val3"], "col2": ["val2", "val4"]})
        mock_read_csv.return_value = mock_df

        # Make to_thread execute the worker function
        mock_to_thread.side_effect = lambda func, *args, **kwargs: func(*args, **kwargs)

        result = await extract_text_from_csv(mock_file)

        # Verify result and mocks
        assert result == expected_text
        mock_file.seek.assert_called_once_with(0)
        mock_file.read.assert_called_once_with()
        mock_read_csv.assert_called_once()
        # Check the argument passed to read_csv is a BytesIO object
        read_csv_arg = mock_read_csv.call_args[0][0]
        assert isinstance(read_csv_arg, BytesIO)
        assert read_csv_arg.read() == csv_content


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

        # Ensure to_thread still calls the worker
        mock_to_thread.side_effect = lambda func, *args, **kwargs: func(*args, **kwargs)

        result = await extract_text_from_csv(mock_file)

        assert result == expected_text_fallback
        mock_file.seek.assert_called_once_with(0)
        mock_file.read.assert_called_once_with()


@pytest.mark.asyncio
async def test_extract_text_from_csv_empty_data_fallback(
    mock_upload_file_factory,
) -> None:
    """Tests CSV parsing fallback for EmptyDataError."""
    csv_content = b""  # Empty content
    mock_file = mock_upload_file_factory("empty.csv", csv_content, "text/csv")
    expected_text_fallback = csv_content.decode(
        "utf-8", errors="replace"
    )  # Which is ""

    with (
        patch(
            "pandas.read_csv", side_effect=EmptyDataError("Simulated empty data error")
        ),
        patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread,
    ):
        mock_to_thread.side_effect = lambda func, *args, **kwargs: func(*args, **kwargs)
        result = await extract_text_from_csv(mock_file)
        assert result == expected_text_fallback  # ""


@pytest.mark.asyncio
async def test_extract_text_from_csv_empty_dataframe(mock_upload_file_factory) -> None:
    """Tests CSV parsing when pandas returns an empty DataFrame."""
    csv_content = b"col1,col2\n"  # Header only
    mock_file = mock_upload_file_factory("header_only.csv", csv_content, "text/csv")

    with (
        patch("pandas.read_csv") as mock_read_csv,
        patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread,
    ):
        import pandas as pd

        mock_df = pd.DataFrame(columns=["col1", "col2"])  # Empty DataFrame
        mock_read_csv.return_value = mock_df
        mock_to_thread.side_effect = lambda func, *args, **kwargs: func(*args, **kwargs)

        result = await extract_text_from_csv(mock_file)
        assert result == ""  # Should return empty string for empty DataFrame


@pytest.mark.asyncio
async def test_extract_text_from_csv_decode_error_fallback(
    mock_upload_file_factory,
) -> None:
    """Tests CSV parsing fallback when decode fails after pandas error."""
    # Use latin-1 content which will fail utf-8 decoding
    csv_content = b"col1,col2\nval1,\xa3"  # \xa3 is pound sign in latin-1
    mock_file = mock_upload_file_factory("latin1.csv", csv_content, "text/csv")
    # Fallback should return empty string if decode fails
    expected_text_fallback = ""

    with (
        patch("pandas.read_csv", side_effect=ParserError("Simulated error")),
        # Mock the decode call within the except block
        patch(
            "builtins.bytes.decode",
            side_effect=UnicodeDecodeError("utf-8", b"", 0, 1, "test"),
        ) as mock_decode,
        patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread,
    ):
        mock_to_thread.side_effect = lambda func, *args, **kwargs: func(*args, **kwargs)
        result = await extract_text_from_csv(mock_file)

        assert result == expected_text_fallback
        # Check that decode was attempted
        mock_decode.assert_called_once_with("utf-8", errors="replace")


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
        # Reading from BytesIO consumes it, so re-create to check content
        assert BytesIO(image_content).read() == image_content

        mock_image_instance.convert.assert_called_once_with("RGB")
        mock_ocr.assert_called_once_with(
            mock_image_instance
        )  # Check it's called with the converted image


@pytest.mark.asyncio
async def test_extract_text_from_image_ocr_error(mock_upload_file_factory) -> None:
    """Tests handling of errors during pytesseract processing."""
    image_content = b"bad_image_bytes"
    mock_file = mock_upload_file_factory("bad.jpg", image_content, "image/jpeg")

    with (
        patch("PIL.Image.open") as mock_pil_open,
        patch(
            "pytesseract.image_to_string", side_effect=Exception("Simulated OCR error")
        ) as mock_ocr,
        patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread,
    ):
        mock_image_instance = MagicMock()
        mock_image_instance.convert.return_value = mock_image_instance
        mock_pil_open.return_value.__enter__.return_value = mock_image_instance

        mock_to_thread.side_effect = lambda func, *args, **kwargs: func(*args, **kwargs)

        result = await extract_text_from_image(mock_file)

        assert "Error extracting image text: Simulated OCR error" in result
        mock_ocr.assert_called_once()


# TXT Parser Tests
@pytest.mark.asyncio
async def test_read_txt_success(mock_upload_file_factory) -> None:
    """Tests successful reading of a UTF-8 text file."""
    txt_content = "Hello, world!\nLine 2.".encode("utf-8")
    mock_file = mock_upload_file_factory("test.txt", txt_content, "text/plain")
    expected_text = "Hello, world!\nLine 2."

    result = await read_txt(mock_file)

    assert result == expected_text
    mock_file.seek.assert_called_once_with(0)
    mock_file.read.assert_called_once_with()


@pytest.mark.asyncio
async def test_read_txt_with_decode_errors(mock_upload_file_factory) -> None:
    """Tests reading a text file with bytes that cannot be decoded as UTF-8."""
    # Mix valid UTF-8 with an invalid byte sequence (e.g., from latin-1)
    txt_content = b"Valid UTF-8 then \xa3 invalid byte"  # \xa3 is invalid in UTF-8
    mock_file = mock_upload_file_factory("mixed.txt", txt_content, "text/plain")
    # The decode uses errors='replace', so invalid bytes become ''
    expected_text = "Valid UTF-8 then invalid byte"

    result = await read_txt(mock_file)

    assert result == expected_text
    mock_file.seek.assert_called_once_with(0)
    mock_file.read.assert_called_once_with()


@pytest.mark.asyncio
async def test_read_txt_empty_file(mock_upload_file_factory) -> None:
    """Tests reading an empty text file."""
    txt_content = b""
    mock_file = mock_upload_file_factory("empty.txt", txt_content, "text/plain")
    expected_text = ""

    result = await read_txt(mock_file)

    assert result == expected_text
    mock_file.seek.assert_called_once_with(0)
    mock_file.read.assert_called_once_with()
