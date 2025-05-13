import pytest
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# stage_filename – simple heuristic stage
# ---------------------------------------------------------------------------
from src.classification.stages.filename import stage_filename
from src.classification.types import ClassificationResult, StageOutcome


@pytest.mark.asyncio
async def test_stage_filename_identifies_invoice() -> None:
    mock_file = MagicMock()
    mock_file.filename = "INV123_invoice.pdf"
    outcome: StageOutcome = await stage_filename(mock_file)
    assert outcome.label == "invoice" and outcome.confidence == pytest.approx(0.85)


@pytest.mark.asyncio
async def test_stage_filename_no_match() -> None:
    mock_file = MagicMock()
    mock_file.filename = "random_file.xyz"
    outcome = await stage_filename(mock_file)
    assert outcome.label is None and outcome.confidence is None


# ---------------------------------------------------------------------------
# src/classifier.classify_file – legacy Flask shim helper
# ---------------------------------------------------------------------------
from src import classifier as _legacy_classifier_mod


@pytest.mark.parametrize(
    "fname, expected",
    [
        ("my_drivers_license_scan.png", "drivers_licence"),
        ("monthly_bank_statement.pdf", "bank_statement"),
        ("vat_invoice_2024.pdf", "invoice"),
        ("mystery.bin", "unknown file"),
    ],
)
def test_legacy_classifier(fname: str, expected: str) -> None:
    dummy = type("Dummy", (), {"filename": fname})()
    assert _legacy_classifier_mod.classify_file(dummy) == expected


# ---------------------------------------------------------------------------
# classification.types.ClassificationResult.dict helper
# ---------------------------------------------------------------------------


def test_classification_result_dict() -> None:
    result = ClassificationResult(
        filename="doc.pdf",
        mime_type="application/pdf",
        size_bytes=1024,
        label="invoice",
        confidence=0.93,
        stage_confidences={"stage_filename": 0.85},
        pipeline_version="v1",
        processing_ms=12.3,
        warnings=[{"msg": "test"}],
        errors=[],
    )

    as_dict = result.dict()
    assert as_dict["filename"] == "doc.pdf"
    assert as_dict["stage_confidences"]["stage_filename"] == 0.85
    assert "warnings" in as_dict and as_dict["warnings"]
