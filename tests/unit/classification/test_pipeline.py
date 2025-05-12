from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from starlette.datastructures import UploadFile

from src.classification.pipeline import ClassificationResult, StageOutcome, classify
from tests.conftest import MockSettings


@pytest.fixture
def mock_upload_file() -> MagicMock:
    """Provides a mock UploadFile object."""
    mock_file = MagicMock(spec=UploadFile)
    mock_file.filename = "test_document.pdf"
    mock_file.content_type = "application/pdf"

    # Mock the file-like object within UploadFile
    mock_file.file = MagicMock()
    mock_file.file.tell.return_value = 12345  # Simulate file size
    mock_file.file.seek = MagicMock()

    # Mock async methods used by the pipeline
    mock_file.seek = AsyncMock()
    mock_file.read = AsyncMock(return_value=b"file content")
    return mock_file


@pytest.fixture
def mock_settings() -> MockSettings:
    """Provides a mock Settings object for pipeline tests."""
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
    mock_stage1.__name__ = "stage_filename"  # Crucial for dict keys
    mock_stage2 = AsyncMock(return_value=mock_stage2_outcome)
    mock_stage2.__name__ = "stage_text"  # Crucial for dict keys

    expected_stage_outcomes_dict = {
        "stage_filename": mock_stage1_outcome,
        "stage_text": mock_stage2_outcome,
    }

    # Patch STAGE_REGISTRY and other dependencies
    with (
        patch("src.classification.pipeline.STAGE_REGISTRY", [mock_stage1, mock_stage2]),
        patch("src.classification.pipeline.get_settings", return_value=mock_settings),
        patch(
            "src.classification.confidence.aggregate_confidences",
            return_value=("invoice", 0.75),  # Mocked final result
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
        # Check processing time approximately
        assert result.processing_ms == pytest.approx(round(processing_ms, 2), abs=50)

        # Assertions for stage_confidences
        assert "stage_filename" in result.stage_confidences
        assert result.stage_confidences["stage_filename"] == 0.8
        assert "stage_text" in result.stage_confidences
        assert result.stage_confidences["stage_text"] == 0.7

        # Assertions for mocks calls
        # Check seek was called before each stage
        mock_upload_file.seek.assert_any_call(0)
        assert mock_upload_file.seek.call_count >= 2  # Called before stage1 and stage2

        mock_stage1.assert_called_once_with(mock_upload_file)
        mock_stage2.assert_called_once_with(mock_upload_file)

        mock_get_size.assert_called_once_with(mock_upload_file)

        # Check arguments to aggregate_confidences
        mock_agg.assert_called_once_with(
            expected_stage_outcomes_dict, settings=mock_settings
        )

        # Assert final logging call matches the updated signature
        mock_logger.info.assert_called_once_with(
            "classification_complete",
            filename="test_document.pdf",
            label="invoice",
            confidence=0.750,
            processing_ms=result.processing_ms,
            pipeline_version="v_test_pipeline",
            stage_outcomes={  # Verify this new logging argument
                "stage_filename": ("invoice", 0.8),
                "stage_text": ("invoice", 0.7),
            },
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
        patch("src.classification.pipeline.STAGE_REGISTRY", []),  # Empty registry
        patch("src.classification.pipeline.get_settings", return_value=mock_settings),
        patch(
            "src.classification.confidence.aggregate_confidences",
            return_value=("unknown", 0.0),
        ) as mock_agg,
        patch("src.classification.pipeline._get_file_size", return_value=500),
        patch("src.classification.pipeline.logger") as mock_logger,  # Mock logger too
    ):

        result = await classify(mock_upload_file)

        assert result.label == "unknown"
        assert result.confidence == 0.0
        assert not result.stage_confidences  # Empty dict
        mock_agg.assert_called_once_with(
            {}, settings=mock_settings
        )  # Called with empty outcomes
        # Check the final log call
        mock_logger.info.assert_called_once_with(
            "classification_complete",
            filename="test_document.pdf",
            label="unknown",
            confidence=0.0,
            processing_ms=result.processing_ms,
            pipeline_version="v_test_pipeline",
            stage_outcomes={},  # empty outcomes
        )


@pytest.mark.asyncio
async def test_classify_unknown_label_from_aggregation(
    mock_upload_file: MagicMock, mock_settings: MockSettings
) -> None:
    """
    Tests when aggregation results in an 'unknown' or low-confidence 'unsure' label.
    """
    mock_stage_outcome = StageOutcome(label=None, confidence=None)
    mock_stage = AsyncMock(return_value=mock_stage_outcome)
    mock_stage.__name__ = "stage_mystery"

    expected_stage_outcomes_dict = {"stage_mystery": mock_stage_outcome}

    with (
        patch("src.classification.pipeline.STAGE_REGISTRY", [mock_stage]),
        patch("src.classification.pipeline.get_settings", return_value=mock_settings),
        patch(
            "src.classification.confidence.aggregate_confidences",
            return_value=("unsure", 0.123),  # Aggregation returns unsure
        ) as mock_agg,
        patch("src.classification.pipeline._get_file_size", return_value=500),
        patch("src.classification.pipeline.logger") as mock_logger,
    ):

        result = await classify(mock_upload_file)

        assert result.label == "unsure"
        assert result.confidence == 0.123  # from mock_agg
        assert "stage_mystery" in result.stage_confidences
        assert result.stage_confidences["stage_mystery"] is None
        mock_agg.assert_called_once_with(
            expected_stage_outcomes_dict, settings=mock_settings
        )
        # Check the final log call
        mock_logger.info.assert_called_once_with(
            "classification_complete",
            filename="test_document.pdf",
            label="unsure",
            confidence=0.123,
            processing_ms=result.processing_ms,
            pipeline_version="v_test_pipeline",
            stage_outcomes={"stage_mystery": (None, None)},
        )


def test_get_file_size_utility(mock_upload_file: MagicMock) -> None:
    """Tests the _get_file_size utility function."""
    # Reset mocks from fixture
    mock_upload_file.file.reset_mock()
    mock_upload_file.file.tell.side_effect = [0, 5678]  # Start pos, End pos after seek

    # Import the private utility for testing
    from src.classification.pipeline import _get_file_size

    size = _get_file_size(mock_upload_file)

    assert size == 5678
    # Check seek calls: to end (0, 2) and back to original (0)
    mock_upload_file.file.seek.assert_any_call(0, 2)
    mock_upload_file.file.seek.assert_any_call(0)
    assert mock_upload_file.file.tell.call_count == 2  # Called to get current and end


@pytest.mark.asyncio
async def test_classify_with_filename_none(mock_settings: MockSettings) -> None:
    """Tests classification when UploadFile.filename is None."""
    mock_file_no_filename = MagicMock(spec=UploadFile)
    mock_file_no_filename.filename = None  # Simulate no filename
    mock_file_no_filename.content_type = "application/octet-stream"
    mock_file_no_filename.file = MagicMock()
    mock_file_no_filename.file.tell.return_value = 100
    mock_file_no_filename.seek = MagicMock()  # Sync mock for file object seek
    mock_file_no_filename.seek = AsyncMock()  # Async mock for UploadFile seek
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
        assert result.filename == "<unknown>"  # Check default filename is used
