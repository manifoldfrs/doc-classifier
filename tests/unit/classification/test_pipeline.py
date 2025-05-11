"""tests/unit/classification/test_pipeline.py
###############################################################################
Unit tests for the classification pipeline orchestrator
(``src.classification.pipeline``).
###############################################################################
These tests verify the core logic of the `classify` function, ensuring it
correctly orchestrates stages, aggregates results, and produces the expected
`ClassificationResult` object.
"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from starlette.datastructures import UploadFile

from src.classification.pipeline import ClassificationResult, StageOutcome, classify
from tests.conftest import MockSettings

# pylint: disable=protected-access


@pytest.fixture
def mock_upload_file() -> MagicMock:
    """Provides a mock UploadFile object."""
    mock_file = MagicMock(spec=UploadFile)
    mock_file.filename = "test_document.pdf"
    mock_file.content_type = "application/pdf"

    # Mock the file-like object within UploadFile
    mock_file.file = MagicMock()
    mock_file.file.tell.return_value = 1024  # Simulate 1KB file size
    mock_file.file.seek = MagicMock()

    # Mock async methods if your UploadFile interactions are async
    mock_file.seek = AsyncMock()
    mock_file.read = AsyncMock(return_value=b"file content")
    return mock_file


@pytest.fixture
def mock_settings() -> MockSettings:
    """Provides a mock Settings object."""
    settings = MockSettings(
        pipeline_version="v_test_pipeline",
        confidence_threshold=0.6,
        early_exit_confidence=0.9,
    )
    return settings


@pytest.mark.asyncio
async def test_classify_successful_execution(
    mock_upload_file: MagicMock, mock_settings: MockSettings
) -> None:
    """
    Tests a successful classification flow with mock stages and confidence aggregation.
    """
    start_time = time.perf_counter()

    # Mock stage outcomes
    mock_stage1_outcome = StageOutcome(label="invoice", confidence=0.8)
    mock_stage2_outcome = StageOutcome(label="invoice", confidence=0.7)

    # Mock stage callables
    mock_stage1 = AsyncMock(return_value=mock_stage1_outcome)
    mock_stage1.__name__ = "stage_filename"
    mock_stage2 = AsyncMock(return_value=mock_stage2_outcome)
    mock_stage2.__name__ = "stage_text"

    # Patch STAGE_REGISTRY and other dependencies
    with (
        patch("src.classification.pipeline.STAGE_REGISTRY", [mock_stage1, mock_stage2]),
        patch("src.classification.pipeline.get_settings", return_value=mock_settings),
        patch(
            "src.classification.confidence.aggregate_confidences",
            return_value=("invoice", 0.75),
        ) as mock_agg,
        patch(
            "src.classification.pipeline._get_file_size", return_value=12345
        ) as mock_get_size,
        patch("src.classification.pipeline.logger") as mock_logger,
    ):

        result = await classify(mock_upload_file)

        end_time = time.perf_counter()
        processing_ms = (end_time - start_time) * 1000

        # Assertions for ClassificationResult
        assert isinstance(result, ClassificationResult)
        assert result.filename == "test_document.pdf"
        assert result.mime_type == "application/pdf"
        assert result.size_bytes == 12345
        assert result.label == "invoice"
        assert result.confidence == 0.750  # from mock_agg
        assert result.pipeline_version == "v_test_pipeline"
        assert result.processing_ms == pytest.approx(
            processing_ms, abs=50
        )  # Allow some leeway

        # Assertions for stage_confidences
        assert "stage_filename" in result.stage_confidences
        assert result.stage_confidences["stage_filename"] == 0.8
        assert "stage_text" in result.stage_confidences
        assert result.stage_confidences["stage_text"] == 0.7

        # Assertions for mocks
        mock_stage1.assert_called_once_with(mock_upload_file)
        mock_stage2.assert_called_once_with(mock_upload_file)

        # Check that _get_file_size was called
        mock_get_size.assert_called_once_with(mock_upload_file)

        # Check arguments to aggregate_confidences
        # The first argument is a dictionary of stage outcomes
        aggregated_outcomes = mock_agg.call_args[0][0]
        assert aggregated_outcomes["stage_filename"] == mock_stage1_outcome
        assert aggregated_outcomes["stage_text"] == mock_stage2_outcome
        assert mock_agg.call_args[1]["settings"] == mock_settings

        # Assert logging
        mock_logger.info.assert_called_once_with(
            "classification_complete",
            filename="test_document.pdf",
            label="invoice",
            confidence=0.750,
            processing_ms=result.processing_ms,  # Use the actual result's processing_ms
            pipeline_version="v_test_pipeline",
        )


@pytest.mark.asyncio
async def test_classify_no_stages_registered(
    mock_upload_file: MagicMock, mock_settings: MockSettings
) -> None:
    """
    Tests pipeline behavior when STAGE_REGISTRY is empty.
    It should default to 'unknown' with 0.0 confidence.
    """
    with (
        patch("src.classification.pipeline.STAGE_REGISTRY", []),
        patch("src.classification.pipeline.get_settings", return_value=mock_settings),
        patch(
            "src.classification.confidence.aggregate_confidences",
            return_value=("unknown", 0.0),
        ) as mock_agg,
        patch("src.classification.pipeline._get_file_size", return_value=500),
    ):

        result = await classify(mock_upload_file)

        assert result.label == "unknown"
        assert result.confidence == 0.0
        assert not result.stage_confidences  # Empty dict
        mock_agg.assert_called_once_with({}, settings=mock_settings)


@pytest.mark.asyncio
async def test_classify_unknown_label_from_aggregation(
    mock_upload_file: MagicMock, mock_settings: MockSettings
) -> None:
    """
    Tests when aggregation results in an 'unknown' label.
    """
    mock_stage = AsyncMock(return_value=StageOutcome(label=None, confidence=None))
    mock_stage.__name__ = "stage_mystery"

    with (
        patch("src.classification.pipeline.STAGE_REGISTRY", [mock_stage]),
        patch("src.classification.pipeline.get_settings", return_value=mock_settings),
        patch(
            "src.classification.confidence.aggregate_confidences",
            return_value=("unknown", 0.123),
        ),
        patch("src.classification.pipeline._get_file_size", return_value=500),
    ):

        result = await classify(mock_upload_file)

        assert result.label == "unknown"
        assert result.confidence == 0.123  # from mock_agg
        assert "stage_mystery" in result.stage_confidences
        assert result.stage_confidences["stage_mystery"] is None


def test_get_file_size_utility(mock_upload_file: MagicMock) -> None:
    """Tests the _get_file_size utility function."""
    # Reset mocks from fixture if they were modified by other tests
    mock_upload_file.file.reset_mock()
    mock_upload_file.file.tell.return_value = 5678  # New specific size

    # Import the private utility for testing
    from src.classification.pipeline import _get_file_size

    # Set up mocks
    # First tell() is called to get current position
    # Second tell() is called to get file size
    mock_upload_file.file.tell.side_effect = [0, 5678]

    size = _get_file_size(mock_upload_file)

    assert size == 5678
    # Seek to end and seek to beginning
    mock_upload_file.file.seek.assert_any_call(0, 2)  # Seek to end
    # We only want to assert it was called with tell(), not the number of calls
    assert mock_upload_file.file.tell.called  # Called at least once
    mock_upload_file.file.seek.assert_any_call(0)  # Seek back to start


@pytest.mark.asyncio
async def test_classify_with_filename_none(mock_settings: MockSettings) -> None:
    """Tests classification when UploadFile.filename is None."""
    mock_file_no_filename = MagicMock(spec=UploadFile)
    mock_file_no_filename.filename = None  # Simulate no filename
    mock_file_no_filename.content_type = "application/octet-stream"
    mock_file_no_filename.file = MagicMock()
    mock_file_no_filename.file.tell.return_value = 100
    mock_file_no_filename.seek = AsyncMock()
    mock_file_no_filename.read = AsyncMock(return_value=b"content")

    with (
        patch("src.classification.pipeline.STAGE_REGISTRY", []),
        patch("src.classification.pipeline.get_settings", return_value=mock_settings),
        patch(
            "src.classification.confidence.aggregate_confidences",
            return_value=("unknown", 0.0),
        ),
        patch("src.classification.pipeline._get_file_size", return_value=100),
    ):

        result = await classify(mock_file_no_filename)
        assert result.filename == "<unknown>"  # Check default filename
