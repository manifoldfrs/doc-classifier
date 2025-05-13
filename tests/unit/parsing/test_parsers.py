from __future__ import annotations

from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch, ANY

import pandas as pd
import pytest
from starlette.datastructures import UploadFile

# Parsers to test
from src.parsing.csv import _dataframe_to_text, extract_text_from_csv
from src.parsing.docx import extract_text_from_docx
from src.parsing.image import extract_text_from_image
from src.parsing.pdf import extract_text_from_pdf
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

    # Patch the extract_text function as it's imported in src.parsing.pdf
    with (
        patch(
            "src.parsing.pdf.extract_text", return_value=expected_text
        ) as mock_extract,
        patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread,
    ):
        # Make to_thread execute the worker function which calls the mocked extract_text
        async def fake_to_thread(worker_fn, *args, **kwargs):
            return worker_fn(*args, **kwargs)

        mock_to_thread.side_effect = fake_to_thread

        result = await extract_text_from_pdf(mock_file)

        # Assertions
        assert result == expected_text
        mock_file.seek.assert_called_once_with(0)
        mock_file.read.assert_called_once_with()
        mock_to_thread.assert_called_once()
        # Check that the (now patched) extract_text was called by the worker
        mock_extract.assert_called_once()
        # Check the first argument passed to extract_text is a BytesIO object
        assert isinstance(mock_extract.call_args[0][0], BytesIO)


@pytest.mark.asyncio
async def test_extract_text_from_pdf_extraction_error(
    mock_upload_file_factory,
) -> None:
    """Tests handling when pdfminer raises specific errors."""
    from pdfminer.pdfdocument import PDFTextExtractionNotAllowed

    pdf_content = b"encrypted pdf"
    mock_file = mock_upload_file_factory("test.pdf", pdf_content, "application/pdf")

    # Patch extract_text in src.parsing.pdf to raise an error
    with (
        patch(
            "src.parsing.pdf.extract_text",
            side_effect=PDFTextExtractionNotAllowed("Extraction denied"),
        ),
        patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread,
    ):

        async def fake_to_thread(worker_fn, *args, **kwargs):
            # The worker function should catch the exception and return ""
            return worker_fn(*args, **kwargs)

        mock_to_thread.side_effect = fake_to_thread

        result = await extract_text_from_pdf(mock_file)
        assert result == ""  # Expect empty string on this specific error


@pytest.mark.asyncio
async def test_extract_text_from_pdf_generic_error(mock_upload_file_factory) -> None:
    """Tests handling of generic Exception during PDF extraction."""
    pdf_content = b"corrupted pdf"
    mock_file = mock_upload_file_factory("test.pdf", pdf_content, "application/pdf")

    # Patch extract_text in src.parsing.pdf to raise a generic Exception
    with (
        patch(
            "src.parsing.pdf.extract_text", side_effect=Exception("Generic PDF error")
        ),
        patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread,
    ):

        async def fake_to_thread(worker_fn, *args, **kwargs):
            # The worker function should catch the generic Exception and return ""
            return worker_fn(*args, **kwargs)

        mock_to_thread.side_effect = fake_to_thread

        result = await extract_text_from_pdf(mock_file)
        assert result == ""  # Expect empty string on generic error


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
async def test_extract_text_from_docx_generic_error(mock_upload_file_factory) -> None:
    """Tests handling of generic Exception during DOCX processing."""
    docx_content = b"bad docx"
    mock_file = mock_upload_file_factory("error.docx", docx_content, "application/docx")
    expected_error_message = "Error extracting DOCX text: Simulated docx2txt error"

    with patch("tempfile.NamedTemporaryFile") as mock_temp_file_constructor:
        mock_temp_file_instance = MagicMock()
        mock_temp_file_instance.name = "dummy_temp_file.docx"
        mock_temp_file_instance.__enter__.return_value = mock_temp_file_instance
        mock_temp_file_instance.__exit__.return_value = None
        mock_temp_file_constructor.return_value = mock_temp_file_instance

        with (
            patch(
                "docx2txt.process", side_effect=Exception("Simulated docx2txt error")
            ) as mock_process,
            patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread,
        ):
            mock_to_thread.side_effect = lambda func, *args, **kwargs: func(
                *args, **kwargs
            )
            result = await extract_text_from_docx(mock_file)

            assert result == expected_error_message
            mock_process.assert_called_once_with("dummy_temp_file.docx")


# CSV Parser Tests
@pytest.mark.asyncio
async def test_extract_text_from_csv_pandas_success(mock_upload_file_factory) -> None:
    """Tests successful CSV parsing using pandas."""
    csv_content = b"col1,col2\nval1,val2"
    mock_file = mock_upload_file_factory("test.csv", csv_content, "text/csv")
    expected_text = "col1 col2\nval1 val2"

    # Mock pandas.read_csv behavior within the worker function
    with (
        patch("pandas.read_csv") as mock_read_csv,
        patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread,
    ):
        # Create a dummy DataFrame to be returned by the mock
        mock_df = pd.DataFrame({"col1": ["val1"], "col2": ["val2"]})
        mock_read_csv.return_value = mock_df

        # Make to_thread execute the worker function
        async def fake_to_thread(worker_fn, *args, **kwargs):
            return worker_fn(*args, **kwargs)

        mock_to_thread.side_effect = fake_to_thread

        result = await extract_text_from_csv(mock_file)

        # Verify result and mocks
        assert result == expected_text
        mock_file.seek.assert_called_once_with(0)
        mock_file.read.assert_called_once_with()
        mock_to_thread.assert_called_once()
        mock_read_csv.assert_called_once()
        # Check the first argument to read_csv is a BytesIO object
        assert isinstance(mock_read_csv.call_args[0][0], BytesIO)


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
            # Worker catches ParserError and returns decoded bytes
            return worker_fn(*args, **kwargs)

        mock_to_thread.side_effect = fake_to_thread

        result = await extract_text_from_csv(mock_file)

        assert result == expected_text_fallback
        mock_file.seek.assert_called_once_with(0)
        mock_file.read.assert_called_once_with()


@pytest.mark.asyncio
async def test_extract_text_from_csv_empty_data_fallback(
    mock_upload_file_factory,
) -> None:
    """Tests CSV parsing fallback when pandas encounters EmptyDataError."""
    csv_content = b""  # Empty content
    mock_file = mock_upload_file_factory("empty.csv", csv_content, "text/csv")
    expected_text_fallback = ""  # Empty string

    with (
        patch(
            "pandas.read_csv", side_effect=EmptyDataError("Simulated empty data error")
        ),
        patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread,
    ):

        async def fake_to_thread(worker_fn, *args, **kwargs):
            return worker_fn(*args, **kwargs)

        mock_to_thread.side_effect = fake_to_thread

        result = await extract_text_from_csv(mock_file)
        assert result == expected_text_fallback


@pytest.mark.asyncio
async def test_extract_text_from_csv_empty_dataframe(
    mock_upload_file_factory,
) -> None:
    """Tests CSV parsing when pandas returns an empty DataFrame."""
    csv_content = b"col1,col2\n"  # Header only, or could be empty content that pandas parses to empty DF
    mock_file = mock_upload_file_factory("empty_df.csv", csv_content, "text/csv")
    expected_text_empty_df = ""  # If DataFrame is empty after read, should return empty

    with (
        patch("pandas.read_csv") as mock_read_csv,
        patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread,
    ):
        mock_df_empty = pd.DataFrame()  # Empty DataFrame
        mock_read_csv.return_value = mock_df_empty

        async def fake_to_thread(worker_fn, *args, **kwargs):
            return worker_fn(*args, **kwargs)

        mock_to_thread.side_effect = fake_to_thread

        result = await extract_text_from_csv(mock_file)
        assert result == expected_text_empty_df


@pytest.mark.asyncio
async def test_extract_text_from_csv_decode_error_fallback(
    mock_upload_file_factory,
) -> None:
    """Tests CSV parsing fallback when pandas fails and content decoding also fails."""
    csv_content_invalid_utf8 = b"col1,col2\nval1,\xffval2"  # \xff is invalid in UTF-8
    mock_file = mock_upload_file_factory(
        "decode_error.csv", csv_content_invalid_utf8, "text/csv"
    )
    # If pandas fails and fallback decoding (with errors='replace') happens:
    expected_text_on_decode_replace = (
        "col1,col2\nval1,\ufffdval2"  # Unicode replacement char
    )

    with (
        patch("pandas.read_csv", side_effect=ParserError("Pandas fails")),
        patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread,
    ):

        async def fake_to_thread(worker_fn, *args, **kwargs):
            return worker_fn(*args, **kwargs)

        mock_to_thread.side_effect = fake_to_thread

        result = await extract_text_from_csv(mock_file)
        assert result == expected_text_on_decode_replace


@pytest.mark.asyncio
async def test_extract_text_from_csv_generic_error(mock_upload_file_factory) -> None:
    """Tests handling of generic Exception during CSV processing."""
    csv_content = b"col1,col2\nval1,val2"
    mock_file = mock_upload_file_factory("error.csv", csv_content, "text/csv")

    with (
        patch(
            "pandas.read_csv", side_effect=Exception("Simulated generic pandas error")
        ),
        patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread,
    ):

        async def fake_to_thread(worker_fn, *args, **kwargs):
            # Worker should catch generic Exception and return ""
            return worker_fn(*args, **kwargs)

        mock_to_thread.side_effect = fake_to_thread

        result = await extract_text_from_csv(mock_file)
        assert result == ""  # Expect empty string on generic error


@pytest.mark.asyncio
async def test_extract_text_from_csv_unicode_decode_error_in_fallback(
    mock_upload_file_factory,
) -> None:
    """Tests CSV parsing fallback with UnicodeDecodeError during bytes.decode()."""
    # Content that will cause UnicodeDecodeError with utf-8
    # (e.g., latin-1 characters misinterpreted as utf-8)
    # For example, b'\xe9' is 'é' in latin-1, but invalid as a standalone byte in utf-8.
    csv_content_bad_encoding = b"col1,col2\nval1,\xe9val2"
    mock_file = mock_upload_file_factory(
        "unicode_error.csv", csv_content_bad_encoding, "text/csv"
    )

    # The fallback attempts content.decode("utf-8", errors="replace")
    # \xe9 will be replaced by \ufffd (REPLACEMENT CHARACTER)
    expected_text_fallback_replaced = "col1,col2\nval1,\ufffdval2"

    with (
        patch("pandas.read_csv", side_effect=ParserError("Pandas fails first")),
        patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread,
        patch("src.parsing.csv.logger") as mock_logger,  # Mock logger
    ):

        async def fake_to_thread(worker_fn, *args, **kwargs):
            # The worker will attempt pandas.read_csv (mocked to fail),
            # then attempt to decode bytes, which itself might log if it replaces chars
            return worker_fn(*args, **kwargs)

        mock_to_thread.side_effect = fake_to_thread

        result = await extract_text_from_csv(mock_file)

        # Expect the string with the replacement character from errors='replace'
        assert result == "col1,col2\nval1,\ufffdval2"
        # Ensure the logger was NOT called for this specific failure path
        mock_logger.warning.assert_not_called()


def test_dataframe_to_text_helper() -> None:
    """Tests the _dataframe_to_text internal helper function."""
    # Basic DataFrame
    df1 = pd.DataFrame({"A": [1, 2], "B": ["x", "y"]})
    assert _dataframe_to_text(df1) == "A B\n1 x\n2 y"

    # DataFrame with NaN
    df2 = pd.DataFrame({"col1": [1.0, None], "col2": ["apple", "banana"]})
    assert _dataframe_to_text(df2) == "col1 col2\n1.0 apple\n banana"

    # DataFrame with different types
    df3 = pd.DataFrame({"int": [1], "float": [3.14], "bool": [True]})
    assert _dataframe_to_text(df3) == "int float bool\n1 3.14 True"

    # Empty DataFrame
    df_empty = pd.DataFrame({"X": [], "Y": []})
    assert _dataframe_to_text(df_empty) == "X Y"  # Only header


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
        # Reading again consumes the BytesIO, so create a new one for assertion
        assert BytesIO(image_content).read() == image_content

        mock_image_instance.convert.assert_called_once_with("RGB")
        mock_ocr.assert_called_once_with(
            mock_image_instance
        )  # Check it's called with the converted image


@pytest.mark.asyncio
async def test_extract_text_from_image_generic_error(mock_upload_file_factory) -> None:
    """Tests handling of generic Exception during image processing."""
    image_content = b"bad image"
    mock_file = mock_upload_file_factory("error.jpg", image_content, "image/jpeg")
    expected_error_msg = "Error extracting image text: Simulated PIL error"

    with (
        patch(
            "PIL.Image.open", side_effect=Exception("Simulated PIL error")
        ) as mock_pil_open,
        patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread,
    ):
        mock_to_thread.side_effect = lambda func, *args, **kwargs: func(*args, **kwargs)
        result = await extract_text_from_image(mock_file)

        assert result == expected_error_msg
        mock_pil_open.assert_called_once()  # Ensure PIL was attempted


# TXT Parser Tests
@pytest.mark.asyncio
async def test_read_txt_success(mock_upload_file_factory) -> None:
    """Tests successful reading of a TXT file."""
    txt_content = "Hello, world!\nThis is a test.".encode("utf-8")
    mock_file = mock_upload_file_factory("test.txt", txt_content, "text/plain")
    expected_text = txt_content.decode("utf-8")

    result = await read_txt(mock_file)

    assert result == expected_text
    mock_file.seek.assert_called_once_with(0)
    mock_file.read.assert_called_once_with()


@pytest.mark.asyncio
async def test_read_txt_decoding_error(mock_upload_file_factory) -> None:
    """Tests reading a TXT file with invalid UTF-8 sequences."""
    # Create bytes that are invalid UTF-8 (e.g., 0x80 is a continuation byte without a start)
    invalid_utf8_content = b"Valid text \x80 invalid sequence"
    mock_file = mock_upload_file_factory(
        "invalid.txt", invalid_utf8_content, "text/plain"
    )
    # Expect replacement character (U+FFFD) where invalid byte was
    expected_text = "Valid text invalid sequence"

    result = await read_txt(mock_file)

    assert result == expected_text
    mock_file.seek.assert_called_once_with(0)
    mock_file.read.assert_called_once_with()
